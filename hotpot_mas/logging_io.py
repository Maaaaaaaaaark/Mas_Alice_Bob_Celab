"""Environment collection and append-only JSONL run logging (spec sec. 15).

``JsonlWriter`` appends one JSON object per line. Existing lines are loaded
up front (one pass over the file) so a re-run skips already-finished
``run_id`` values instead of duplicating them; a partially-written trailing
line from an interrupted run is discarded.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Set


def collect_environment_info() -> Dict[str, Any]:
    """Collect software/hardware info once per process (spec sec. 15.1)."""
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import numpy

        info["numpy_version"] = numpy.__version__
    except Exception:  # noqa: BLE001 - environment info is best-effort
        info["numpy_version"] = None
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            )
        else:
            info["gpu_name"] = None
            info["gpu_memory_gb"] = None
    except Exception:  # noqa: BLE001
        info["torch_version"] = None
        info["cuda_available"] = None
        info["cuda_version"] = None
        info["gpu_name"] = None
        info["gpu_memory_gb"] = None
    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except Exception:  # noqa: BLE001
        info["transformers_version"] = None
    try:
        import datasets

        info["datasets_version"] = datasets.__version__
    except Exception:  # noqa: BLE001
        info["datasets_version"] = None
    return info


class JsonlWriter:
    """Append-only JSONL writer with resume support (deduplicates run_ids)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.existing_ids: Set[str] = set()
        if self.path.is_file():
            self.existing_ids = self._load_existing_ids()

    def _load_existing_ids(self) -> Set[str]:
        ids: Set[str] = set()
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    run_id = record.get("run_id")
                    if isinstance(run_id, str):
                        ids.add(run_id)
                except json.JSONDecodeError:
                    # Trailing partial line from an interrupted run: ignore.
                    continue
        return ids

    def contains(self, run_id: str) -> bool:
        return run_id in self.existing_ids

    def append(self, record: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.existing_ids.add(record["run_id"])


def load_runs(path: Path) -> List[Dict[str, Any]]:
    """Load all complete run records from a JSONL file."""
    runs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs
