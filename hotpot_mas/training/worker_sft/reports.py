"""Report writers for Stage 1a worker SFT data preparation.

Three reports are written by ``prepare-worker-sft-data``:

- ``label_quality_report.json``: per-source counts by label/status,
  every excluded example with its reason, and the separate review
  status column (auto checks never mark anything reviewed).
- ``dedup_report.json``: produced by the dedup pipeline.
- ``data_manifest.json``: source availability, dataset provenance,
  mixture summary, and file hashes so the prepared data can be traced
  back to its origins.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .checks import is_reviewed
from .convert import WorkerExample


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def label_quality_report(
    examples: List[WorkerExample],
    teacher_status: Dict[str, str],
    reviewed_problems: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Per-source label/status counts + exclusions + review separation."""
    per_source: Dict[str, Any] = {}
    exclusions: List[Dict[str, str]] = []
    for example in examples:
        entry = per_source.setdefault(
            example.source,
            {
                "positive": 0,
                "negative": 0,
                "reviewed": 0,
                "auto_checked": 0,
                "excluded": 0,
            },
        )
        if example.excluded:
            entry["excluded"] += 1
            exclusions.append(
                {
                    "example_id": example.example_id,
                    "reason": example.exclusion_reason or "unknown",
                }
            )
            continue
        if example.answerable:
            entry["positive"] += 1
        else:
            entry["negative"] += 1
        if is_reviewed(example):
            entry["reviewed"] += 1
        else:
            entry["auto_checked"] += 1
    return {
        "per_source": per_source,
        "exclusions": exclusions,
        "review_policy": (
            "review status is separate from automatic checks: an example "
            "that passes all automatic checks is auto_checked unless a "
            "human reviewer marked it reviewed with a recorded method and "
            "provenance"
        ),
        "teacher_file": teacher_status,
        "reviewed_file_problems": reviewed_problems,
    }


def data_manifest(
    sources: Dict[str, Dict[str, Any]],
    mixture: Dict[str, Any],
    dedup_counts: Dict[str, Any],
    prepared_files: Dict[str, Path],
    config: Any,
) -> Dict[str, Any]:
    """Availability + provenance + mixture + hashes of prepared data."""
    files = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in prepared_files.items()
    }
    return {
        "sources": sources,
        "mixture": mixture,
        "dedup_counts": dedup_counts,
        "prepared_files": files,
        "config": config.to_dict(),
        "note": (
            "the answer and aliases live only in example metadata and are "
            "never rendered into prompts; original document text is stored "
            "untouched"
        ),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def write_examples_jsonl(path: Path, examples: List[WorkerExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(
                json.dumps(example.to_dict(), ensure_ascii=False) + "\n"
            )
