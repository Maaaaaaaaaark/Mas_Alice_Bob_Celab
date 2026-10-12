"""Data preparation runner for Stage 1a worker SFT.

``prepare_worker_sft_data`` loads the three official sources, converts
auto candidates, resolves the reviewed (and optional teacher) entries
against official rows, runs automatic label checks, runs the dedup
pipeline (exclusion file -> eval manifests -> frozen val/test ->
cross-split isolation -> within-train dedup), selects the training
mixture for the requested mode, and writes everything under
``outputs/worker_sft_data/<experiment_version>/``:

- ``candidates.jsonl``: every converted example with checks/review status
- ``prepared_train.jsonl`` / ``prepared_val.jsonl`` /
  ``prepared_test.jsonl`` / ``probes.jsonl``
- ``label_quality_report.json``, ``dedup_report.json``,
  ``data_manifest.json``

The exclusion-file sha256 is verified before anything else. Missing
data sources are hard errors (except the optional teacher file, whose
absence is reported in the manifest).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .checks import checked_example
from .config import WorkerSFTConfig
from .convert import (
    convert_musique_row,
    convert_squad2_row,
    load_reviewed_file,
    load_teacher_file,
)
from .dedup import (
    load_exclusion_index,
    run_dedup_pipeline,
    verify_file_sha256,
)
from .mix import (
    select_smoke_or_normal,
    select_trace,
    source_mixture_report,
)
from .reports import (
    data_manifest,
    label_quality_report,
    write_examples_jsonl,
    write_json,
)
from .sources import (
    build_reviewed_loaders,
    index_rows_by_id,
    load_hotpotqa_train,
    load_musique,
    load_squad2,
)

logger = logging.getLogger("hotpot_mas.training.worker_sft.runner")


def prepare_worker_sft_data(
    cfg: WorkerSFTConfig,
    mode: str,
    force: bool = False,
) -> Dict[str, Path]:
    """Run the full data-prep pipeline for one config/mode."""
    data_dir = cfg.prepared_data_dir()
    candidates_path = data_dir / "candidates.jsonl"
    if data_dir.is_dir() and any(data_dir.iterdir()) and not force:
        raise FileExistsError(
            f"prepared data already exists at {data_dir}; pass --force to "
            "rebuild it (no silent overwrite)"
        )
    if force and data_dir.is_dir():
        import shutil

        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Exclusion file (verified before anything else).
    sha_ok, sha_actual = verify_file_sha256(
        cfg.data.exclusion_file, cfg.data.exclusion_sha256
    )
    if not sha_ok:
        raise ValueError(
            f"exclusion file sha256 mismatch: expected "
            f"{cfg.data.exclusion_sha256}, got {sha_actual} "
            f"({cfg.data.exclusion_file})"
        )
    exclusion_index = load_exclusion_index(cfg.data.exclusion_file)

    # 2. Official sources. Enabled-but-failed sources are hard errors.
    sources_report: Dict[str, Dict[str, Any]] = {}
    loaders: Dict[str, Any] = {}
    all_examples: List[Any] = []

    squad_rows: List[Dict[str, Any]] = []
    musique_rows: List[Dict[str, Any]] = []
    hotpot_rows: List[Dict[str, Any]] = []
    if cfg.data.sources["squad2"].enabled:
        squad_rows, squad_provenance = load_squad2(
            cfg.data.sources["squad2"].official_split
        )
        sources_report["squad2"] = {
            "status": "loaded", "rows": len(squad_rows),
            "provenance": squad_provenance,
        }
        cap = cfg.data.sources["squad2"].max_candidates
        squad_count = 0
        for row in squad_rows:
            if cap and squad_count >= cap:
                break
            converted = convert_squad2_row(row, dict(squad_provenance))
            all_examples.extend(converted)
            squad_count += len(converted)
    else:
        sources_report["squad2"] = {"status": "disabled", "rows": 0}

    if cfg.data.sources["musique"].enabled:
        musique_rows, musique_provenance = load_musique(
            cfg.data.sources["musique"].official_split
        )
        sources_report["musique"] = {
            "status": "loaded", "rows": len(musique_rows),
            "provenance": musique_provenance,
        }
        cap = cfg.data.sources["musique"].max_candidates
        musique_count = 0
        for row in musique_rows:
            if cap and musique_count >= cap:
                break
            converted = convert_musique_row(row, dict(musique_provenance))
            remaining = cap - musique_count if cap else len(converted)
            converted = converted[:remaining]
            all_examples.extend(converted)
            musique_count += len(converted)
    else:
        sources_report["musique"] = {"status": "disabled", "rows": 0}

    if cfg.data.sources["hotpot"].enabled:
        hotpot_rows, hotpot_provenance = load_hotpotqa_train()
        sources_report["hotpot"] = {
            "status": "loaded", "rows": len(hotpot_rows),
            "provenance": hotpot_provenance,
            "training_example_source": (
                "reviewed/teacher files only; no live teacher call, no auto "
                "candidates from HotpotQA"
            ),
        }
    else:
        sources_report["hotpot"] = {"status": "disabled", "rows": 0}
    loaders = build_reviewed_loaders(squad_rows, musique_rows, hotpot_rows)

    # 3. Reviewed file (hard error if it cannot be resolved at all).
    reviewed_examples, reviewed_problems = load_reviewed_file(
        cfg.data.reviewed_file, loaders
    )
    if not reviewed_examples:
        raise ValueError(
            "no reviewed examples could be resolved from "
            f"{cfg.data.reviewed_file}: {reviewed_problems}"
        )
    all_examples.extend(reviewed_examples)

    # 4. Optional teacher file (absence reported, never fatal).
    teacher_entries, teacher_status = load_teacher_file(
        cfg.data.teacher_file
    ) if cfg.data.teacher_file else ([], {"status": "disabled", "path": None})
    # Teacher entries are resolved against official rows here so that
    # any file dropped in later follows the same verification path.
    teacher_examples: List[Any] = []
    teacher_problems: List[Dict[str, str]] = []
    if teacher_entries:
        from .convert import resolve_reviewed_entry

        for entry in teacher_entries:
            example, problem = resolve_reviewed_entry(entry, loaders)
            if problem is not None:
                teacher_problems.append(
                    {"example_id": str(entry.get("example_id", "?")),
                     "reason": problem}
                )
                continue
            # Teacher-generated entries are auto-checked candidates until
            # a human marks them reviewed in the file itself.
            example.review = dict(entry.get("review") or {})
            example.review.setdefault("status", "auto_checked")
            example.review["provenance_file"] = str(cfg.data.teacher_file)
            teacher_examples.append(example)
        all_examples.extend(teacher_examples)
    teacher_status["problems"] = teacher_problems

    # 5. Automatic label checks (review status kept separate).
    for example in all_examples:
        checked_example(example)
    # The rendered-target format check needs the prompting module; run
    # it here for every non-excluded example.
    from .checks import example_matches_target_format
    from .prompting import build_messages, load_instruction

    instruction = load_instruction(cfg.prompt_dir)
    for example in all_examples:
        if example.excluded:
            continue
        _, target = build_messages(example, instruction, 0)
        ok, reason = example_matches_target_format(example, target)
        example.checks["format_ok"] = {"ok": ok, "reason": reason}
        if not ok and not example.excluded:
            example.excluded = True
            example.exclusion_reason = (
                "failed automatic label checks: "
                f"format_ok: {reason}"
            )

    # 6. Dedup pipeline. Reviewed parents are protected from the frozen
    # val/test in every mode so the reviewed set stays available for
    # training; val/test then come from auto candidates.
    reviewed_parents: Set[str] = {
        e.parent_id for e in reviewed_examples
    }
    after_dedup, dedup_report = run_dedup_pipeline(
        all_examples,
        exclusion_index,
        cfg.data.eval_manifest_paths,
        val_parents=cfg.mix.val_parents,
        test_parents=cfg.mix.test_parents,
        freeze_seed=cfg.mix.val_test_freeze_seed,
        question_jaccard_threshold=cfg.mix.near_dup_question_jaccard,
        paragraph_jaccard_threshold=cfg.mix.near_dup_paragraph_jaccard,
        avoid_parent_ids=reviewed_parents,
    )

    # 7. Mixture selection for the requested mode.
    require_reviewed = cfg.review.require_reviewed_for_training
    if mode == "trace":
        train_selected, _ = select_trace(after_dedup, cfg)
        mixture_report = source_mixture_report(train_selected)
        mixture_report["selection"] = "trace"
    else:
        train_selected, mixture_report = select_smoke_or_normal(
            after_dedup, cfg, require_reviewed=require_reviewed
        )
        mixture_report["selection"] = mode

    # 8. Write everything.
    probes = [
        e
        for e in after_dedup
        if e.probe and not e.excluded
    ]
    val_examples = [
        e
        for e in after_dedup
        if e.split == "val" and not e.excluded and not e.probe
    ]
    test_examples = [
        e
        for e in after_dedup
        if e.split == "test" and not e.excluded and not e.probe
    ]

    write_examples_jsonl(candidates_path, after_dedup)
    prepared_files: Dict[str, Path] = {"candidates": candidates_path}
    for name, examples in (
        ("prepared_train", train_selected),
        ("prepared_val", val_examples),
        ("prepared_test", test_examples),
        ("probes", probes),
    ):
        path = data_dir / f"{name}.jsonl"
        write_examples_jsonl(path, examples)
        prepared_files[name] = path

    quality_report = label_quality_report(
        after_dedup, teacher_status, reviewed_problems
    )
    quality_path = data_dir / "label_quality_report.json"
    write_json(quality_path, quality_report)

    dedup_path = data_dir / "dedup_report.json"
    write_json(dedup_path, dedup_report)

    manifest = data_manifest(
        sources_report,
        mixture_report,
        dedup_report["counts"],
        prepared_files,
        cfg,
    )
    manifest["dedup"] = dedup_report
    manifest_path = data_dir / "data_manifest.json"
    write_json(manifest_path, manifest)

    splits = {
        "experiment_version": cfg.experiment_version,
        "mode": mode,
        "seed": cfg.seed,
        "train_example_ids": [e.example_id for e in train_selected],
        "val_example_ids": [e.example_id for e in val_examples],
        "test_example_ids": [e.example_id for e in test_examples],
        "probe_example_ids": [e.example_id for e in probes],
        "reviewed_file_sha256": _sha256_text(
            cfg.data.reviewed_file.read_text(encoding="utf-8")
        ),
    }
    splits_path = data_dir / "splits.json"
    write_json(splits_path, splits)

    logger.info("mixture: %s", mixture_report)
    logger.info("dedup counts: %s", dedup_report["counts"])
    logger.info(
        "reviewed file problems: %d", len(reviewed_problems)
    )
    return prepared_files


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prepared_examples(
    cfg: WorkerSFTConfig, names: Tuple[str, ...] = (
        "prepared_train", "prepared_val", "prepared_test", "probes"
    )
) -> Dict[str, List[Any]]:
    """Load the prepared example files for training."""
    from .convert import WorkerExample

    data_dir = cfg.prepared_data_dir()
    loaded: Dict[str, List[Any]] = {}
    for name in names:
        path = data_dir / f"{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"prepared data missing: {path}; run "
                "prepare-worker-sft-data first"
            )
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(WorkerExample.from_dict(json.loads(line)))
        loaded[name] = examples
    return loaded
