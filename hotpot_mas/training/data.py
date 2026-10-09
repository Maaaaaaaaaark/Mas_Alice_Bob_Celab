"""Training data preparation: three fixed manifests plus ``splits.json``.

The pipeline selects questions from the official HotpotQA distractor
context with the ``balanced_distractor`` partition (one supporting document
plus four distractors per worker). This is an **oracle-balanced partition**:
it uses the gold supporting-document metadata that ships with HotpotQA to
decide which documents are "supporting". It is not an agent assignment that
HotpotQA distributes officially; that fact is recorded here, in
``splits.json`` and in ``docs/TRAINING.md``.

Split construction (deterministic, fully config-controlled):

- ``train``: selected from the configured official split with
  ``train.selection_seed``.
- ``val``: selected from the configured official split with
  ``val.selection_seed``, excluding every train question id.
- ``test``: selected from the configured official split with
  ``test.selection_seed``, excluding every train and val question id.

The val/test boundary is therefore drawn by sequential exclusion under the
per-split selection seeds (``val_test_split_seed`` is recorded in
``splits.json`` as provenance identifying this split configuration; it does
not affect the selection math, which is fully determined by the split
sources, counts and selection seeds). Pairwise disjointness of the three
question id sets is asserted before ``splits.json`` is written.

Nothing here is used by the inference baseline; that pipeline keeps its own
manifests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from hotpot_mas.hotpotqa import load_hotpotqa_validation
from hotpot_mas.question_selection import (
    SelectedQuestion,
    build_manifest,
    load_manifest,
)

from .config import TrainingConfig

SPLITS_FILENAME = "splits.json"

ORACLE_PARTITION_NOTE = (
    "oracle-balanced partition: worker evidence assignment uses the gold "
    "supporting-document metadata shipped with HotpotQA; this is not an "
    "official HotpotQA agent split"
)


def manifest_path_for_split(cfg: TrainingConfig, split: str) -> Path:
    """Manifest file path for one of ``train``/``val``/``test``."""
    return cfg.manifest_dir / f"{split}.json"


def splits_path(cfg: TrainingConfig) -> Path:
    return cfg.manifest_dir / SPLITS_FILENAME


def _source_rows(cfg: TrainingConfig) -> Dict[str, List[Dict[str, Any]]]:
    """Load each distinct official source split at most once."""
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split_name in {
        cfg.data.train.split,
        cfg.data.val.split,
        cfg.data.test.split,
    }:
        if split_name not in rows_by_split:
            rows_by_split[split_name] = load_hotpotqa_validation(
                dataset_config=cfg.data.dataset_config,
                dataset_name=cfg.data.dataset,
                split=split_name,
            )
    return rows_by_split


def _manifest_ids(manifest_path: Path) -> List[str]:
    return [q.question_id for q in load_manifest(manifest_path)]


def assert_split_isolation(
    ids_by_split: Dict[str, List[str]],
) -> None:
    """Raise if any two splits share a question id, or a split has dupes."""
    seen: Set[str] = set()
    for split_name, ids in ids_by_split.items():
        if len(set(ids)) != len(ids):
            raise RuntimeError(
                f"split {split_name!r} contains duplicate question ids"
            )
        for qid in ids:
            if qid in seen:
                raise RuntimeError(
                    f"question {qid!r} appears in more than one split; "
                    f"splits must be pairwise disjoint"
                )
            seen.add(qid)


def prepare_manifests(
    cfg: TrainingConfig, force: bool = False
) -> Dict[str, Path]:
    """Build (or reuse) the three split manifests and ``splits.json``.

    Returns ``{split: manifest_path}``. With ``force=False`` an existing,
    valid ``splits.json`` is reused as-is; with ``force=True`` everything is
    rebuilt from the source splits.
    """
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    sp = splits_path(cfg)
    if sp.exists() and not force:
        loaded = load_splits(cfg.manifest_dir)
        assert_split_isolation(
            {
                name: loaded["splits"][name]["question_ids"]
                for name in ("train", "val", "test")
            }
        )
        return {
            name: manifest_path_for_split(cfg, name)
            for name in ("train", "val", "test")
        }

    rows_by_split = _source_rows(cfg)
    paths: Dict[str, Path] = {}
    ids_by_split: Dict[str, List[str]] = {}

    # Train first: nothing to exclude.
    train_path = manifest_path_for_split(cfg, "train")
    build_manifest(
        rows_by_split[cfg.data.train.split],
        cfg.data.train.num_questions,
        cfg.data.train.selection_seed,
        train_path,
        evidence_partition="balanced_distractor",
        partition_seed=cfg.data.partition_seed,
    )
    paths["train"] = train_path
    ids_by_split["train"] = _manifest_ids(train_path)

    # Val excludes every train id; test excludes train and val ids.
    exclude = set(ids_by_split["train"])
    val_path = manifest_path_for_split(cfg, "val")
    build_manifest(
        rows_by_split[cfg.data.val.split],
        cfg.data.val.num_questions,
        cfg.data.val.selection_seed,
        val_path,
        evidence_partition="balanced_distractor",
        partition_seed=cfg.data.partition_seed,
        exclude_ids=set(exclude),
    )
    paths["val"] = val_path
    ids_by_split["val"] = _manifest_ids(val_path)
    exclude.update(ids_by_split["val"])

    test_path = manifest_path_for_split(cfg, "test")
    build_manifest(
        rows_by_split[cfg.data.test.split],
        cfg.data.test.num_questions,
        cfg.data.test.selection_seed,
        test_path,
        evidence_partition="balanced_distractor",
        partition_seed=cfg.data.partition_seed,
        exclude_ids=set(exclude),
    )
    paths["test"] = test_path
    ids_by_split["test"] = _manifest_ids(test_path)

    assert_split_isolation(ids_by_split)

    split_entries: Dict[str, Any] = {}
    for name in ("train", "val", "test"):
        split_cfg = getattr(cfg.data, name)
        manifest = json.loads(paths[name].read_text(encoding="utf-8"))
        split_entries[name] = {
            "manifest": str(paths[name]),
            "source_split": split_cfg.split,
            "num_questions": split_cfg.num_questions,
            "selection_seed": split_cfg.selection_seed,
            "question_ids": ids_by_split[name],
            "questions_sha256": manifest["questions_sha256"],
            "excluded_statistics": manifest["excluded_statistics"],
            "exclusion_note": (
                "val excludes all train question ids; test excludes all "
                "train and val question ids"
                if name != "train"
                else "no exclusions (train is selected first)"
            ),
        }

    identity_payload = json.dumps(
        {
            name: split_entries[name]["question_ids"]
            for name in ("train", "val", "test")
        },
        sort_keys=True,
    )
    splits = {
        "schema_version": 1,
        "dataset": cfg.data.dataset,
        "dataset_config": cfg.data.dataset_config,
        "partition_seed": cfg.data.partition_seed,
        "val_test_split_seed": cfg.data.val_test_split_seed,
        "partition_note": ORACLE_PARTITION_NOTE,
        "val_test_split_note": (
            "val/test boundary drawn by sequential exclusion under the "
            "per-split selection seeds; val_test_split_seed is recorded "
            "as provenance for this split configuration"
        ),
        "splits": split_entries,
        "splits_disjoint": True,
        "splits_sha256": hashlib.sha256(
            identity_payload.encode("utf-8")
        ).hexdigest(),
    }
    sp.write_text(
        json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return paths


def load_splits(manifest_dir: Path) -> Dict[str, Any]:
    """Load ``splits.json`` and verify every referenced manifest.

    Verification covers: the file exists, the recorded question id list
    matches the manifest contents, and the manifest's own payload hash
    (checked inside :func:`load_manifest`).
    """
    sp = manifest_dir / SPLITS_FILENAME
    if not sp.exists():
        raise FileNotFoundError(
            f"splits.json not found in {manifest_dir}; run "
            f"`prepare-data` first"
        )
    splits = json.loads(sp.read_text(encoding="utf-8"))
    ids_by_split: Dict[str, List[str]] = {}
    for name, entry in splits["splits"].items():
        manifest_path = Path(entry["manifest"])
        if not manifest_path.is_absolute():
            manifest_path = manifest_dir / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest for split {name!r} missing: {manifest_path}"
            )
        ids = _manifest_ids(manifest_path)
        if ids != entry["question_ids"]:
            raise RuntimeError(
                f"splits.json question ids for {name!r} do not match the "
                f"manifest {manifest_path}"
            )
        ids_by_split[name] = ids
    assert_split_isolation(ids_by_split)
    identity_payload = json.dumps(ids_by_split, sort_keys=True)
    digest = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
    if digest != splits["splits_sha256"]:
        raise RuntimeError(
            "splits.json integrity check failed (question id payload "
            "hash mismatch)"
        )
    return splits


def load_questions_for_split(
    cfg: TrainingConfig, split: str
) -> List[SelectedQuestion]:
    """Load one split's selected questions from its manifest."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}")
    return load_manifest(manifest_path_for_split(cfg, split))


def describe_split(cfg: TrainingConfig) -> str:
    """Human-readable split report, printed before every training run."""
    lines = [
        "Data split configuration:",
        f"  dataset: {cfg.data.dataset} ({cfg.data.dataset_config})",
        f"  partition: {ORACLE_PARTITION_NOTE}",
        (
            f"  partition_seed={cfg.data.partition_seed}, "
            f"val_test_split_seed={cfg.data.val_test_split_seed}"
        ),
    ]
    for name in ("train", "val", "test"):
        split_cfg = getattr(cfg.data, name)
        lines.append(
            f"  {name:5s}: official split={split_cfg.split}, "
            f"num_questions={split_cfg.num_questions}, "
            f"selection_seed={split_cfg.selection_seed}"
        )
    lines.append(
        "  val/test boundary: val selected first (excluding train ids), "
        "then test (excluding train and val ids); disjointness asserted"
    )
    lines.append(
        "  official test split has no public gold answers and is not used "
        "as the local test set"
    )
    return "\n".join(lines)
