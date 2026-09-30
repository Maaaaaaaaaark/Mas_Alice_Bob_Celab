"""Deterministic question sharding and safe JSONL shard merging."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from .config import ExperimentConfig
from .question_selection import load_manifest
from .report import write_report


def validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index}"
        )


def load_questions_for_config(cfg: ExperimentConfig) -> List[Dict[str, Any]]:
    """Load the configured manifest prefix with legacy hash compatibility."""
    questions: List[Dict[str, Any]] = []
    for selected in load_manifest(cfg.manifest_path):
        question = asdict(selected)
        if not question.get("evidence_all"):
            question.pop("evidence_all", None)
        questions.append(question)
    if len(questions) < cfg.num_questions:
        raise RuntimeError(
            f"manifest has {len(questions)} questions but config requires "
            f"{cfg.num_questions}"
        )
    return questions[: cfg.num_questions]


def select_question_shard(
    questions: List[Dict[str, Any]], num_shards: int, shard_index: int
) -> List[Dict[str, Any]]:
    """Round-robin questions into stable, disjoint, balanced shards."""
    validate_shard_args(num_shards, shard_index)
    return questions[shard_index::num_shards]


def shard_name(num_shards: int, shard_index: int) -> str:
    validate_shard_args(num_shards, shard_index)
    width = max(3, len(str(num_shards - 1)))
    return f"shard-{shard_index:0{width}d}-of-{num_shards:0{width}d}"


def shard_output_dir(
    cfg: ExperimentConfig, num_shards: int, shard_index: int
) -> Path:
    return cfg.output_subdir() / "shards" / shard_name(num_shards, shard_index)


def _expected_runs(
    questions: Iterable[Dict[str, Any]], run_seeds: Iterable[int]
) -> Dict[str, Tuple[str, int]]:
    return {
        f"{question['question_id']}-seed-{seed}": (
            str(question["question_id"]),
            int(seed),
        )
        for question in questions
        for seed in run_seeds
    }


def _condition_signature(record: Dict[str, Any]) -> str:
    """Hash condition-level metadata while excluding host and run identity."""
    environment = record.get("environment", {})
    payload = {
        "config": record.get("config"),
        "prompt_hashes": record.get("prompt_hashes"),
        "engine_info": record.get("engine_info"),
        "environment": {
            key: environment.get(key)
            for key in (
                "python_version",
                "platform",
                "numpy_version",
                "torch_version",
                "cuda_version",
                "gpu_name",
                "gpu_memory_gb",
                "transformers_version",
                "datasets_version",
            )
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_jsonl_ids(
    path: Path,
    expected_runs: Dict[str, Tuple[str, int]],
    cfg: ExperimentConfig,
    num_shards: int,
    shard_index: int,
) -> Tuple[Set[str], Set[str]]:
    """Validate every JSONL line without retaining large records in memory."""
    if not path.is_file():
        raise RuntimeError(f"missing shard file: {path}")
    seen: Set[str] = set()
    condition_signatures: Set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            run_id = record.get("run_id")
            if not isinstance(run_id, str):
                raise RuntimeError(
                    f"missing run_id in {path}:{line_number}"
                )
            if run_id in seen:
                raise RuntimeError(f"duplicate run_id {run_id!r} in {path}")
            if run_id not in expected_runs:
                raise RuntimeError(
                    f"unexpected run_id {run_id!r} in {path}; shard assignment "
                    "or config does not match"
                )
            expected_question_id, expected_seed = expected_runs[run_id]
            if str(record.get("question_id")) != expected_question_id or int(
                record.get("run_seed", -1)
            ) != expected_seed:
                raise RuntimeError(
                    f"run identity fields disagree with run_id {run_id!r} "
                    f"in {path}"
                )
            if record.get("experiment_id") != cfg.experiment_id or record.get(
                "experiment_version"
            ) != cfg.experiment_version:
                raise RuntimeError(
                    f"experiment identity mismatch for {run_id!r} in {path}"
                )
            if not record.get("run_fingerprint"):
                raise RuntimeError(
                    f"missing run_fingerprint for {run_id!r} in {path}"
                )
            condition_signatures.add(_condition_signature(record))
            shard = record.get("shard")
            if shard is not None and shard != {
                "num_shards": num_shards,
                "shard_index": shard_index,
            }:
                raise RuntimeError(
                    f"shard metadata mismatch for {run_id!r} in {path}"
                )
            seen.add(run_id)
    missing = set(expected_runs) - seen
    if missing:
        examples = sorted(missing)[:5]
        raise RuntimeError(
            f"shard {shard_index} is incomplete: missing {len(missing)} runs; "
            f"examples: {examples}"
        )
    return seen, condition_signatures


def merge_shard_outputs(
    cfg: ExperimentConfig, num_shards: int
) -> Dict[str, Any]:
    """Validate complete shards, atomically merge them, and write reports."""
    validate_shard_args(num_shards, 0)
    if num_shards == 1:
        raise ValueError("merge-shards requires num_shards greater than 1")

    questions = load_questions_for_config(cfg)
    shard_paths: List[Path] = []
    all_seen: Set[str] = set()
    per_shard_counts: List[int] = []
    condition_signatures: Set[str] = set()

    for shard_index in range(num_shards):
        shard_questions = select_question_shard(
            questions, num_shards, shard_index
        )
        expected = _expected_runs(shard_questions, cfg.run_seeds)
        path = shard_output_dir(
            cfg, num_shards, shard_index
        ) / "runs.jsonl"
        seen, shard_signatures = _strict_jsonl_ids(
            path, expected, cfg, num_shards, shard_index
        )
        overlap = all_seen & seen
        if overlap:
            raise RuntimeError(
                f"run IDs occur in multiple shards: {sorted(overlap)[:5]}"
            )
        all_seen.update(seen)
        condition_signatures.update(shard_signatures)
        per_shard_counts.append(len(seen))
        shard_paths.append(path)

    expected_all = _expected_runs(questions, cfg.run_seeds)
    if all_seen != set(expected_all):
        raise RuntimeError(
            "merged shard ID set does not match the configured experiment"
        )
    if len(condition_signatures) != 1:
        raise RuntimeError(
            "shards contain inconsistent config, prompt, engine, software, "
            "or GPU metadata; refusing to merge different conditions"
        )

    output_dir = cfg.output_subdir()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "runs.jsonl"
    temp_path = output_dir / ".runs.jsonl.merge-tmp"
    digest = hashlib.sha256()
    try:
        with open(temp_path, "wb") as destination:
            for path in shard_paths:
                with open(path, "rb") as source:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        destination.write(chunk)
                        digest.update(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temp_path, merged_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    report_paths = write_report(merged_path, output_dir)
    summary = {
        "experiment_id": cfg.experiment_id,
        "experiment_version": cfg.experiment_version,
        "num_shards": num_shards,
        "num_questions": len(questions),
        "run_seeds": list(cfg.run_seeds),
        "expected_runs": len(expected_all),
        "merged_runs": len(all_seen),
        "per_shard_run_counts": per_shard_counts,
        "runs_sha256": digest.hexdigest(),
        "condition_signature": next(iter(condition_signatures)),
        "merged_runs_path": str(merged_path),
        "shard_paths": [str(path) for path in shard_paths],
        "report_paths": {
            name: str(path) for name, path in report_paths.items()
        },
    }
    summary_path = output_dir / "merge_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary["merge_summary_path"] = str(summary_path)
    return summary
