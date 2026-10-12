"""Tokenizer-only length inspection for Stage 1a prepared data.

Loads the prepared examples and the TOKENIZER only (no model weights, no
training), builds each example's prompt + target with the native chat
template, and writes a length report. Anything over
``max_input_tokens`` is listed explicitly — the trainer treats it as a
hard error, and this report is how you inspect lengths BEFORE a run so
you can raise the limit deliberately instead of truncating silently.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .config import WorkerSFTConfig
from .prompting import build_messages, load_instruction
from .runner import load_prepared_examples

logger = logging.getLogger("hotpot_mas.training.worker_sft.inspect")


def inspect_worker_sft_lengths(cfg: WorkerSFTConfig) -> Path:
    """Write ``lengths_report.json`` into the prepared-data dir."""
    from transformers import AutoTokenizer

    from .masking import build_worker_sft_example

    prepared = load_prepared_examples(cfg)
    instruction = load_instruction(cfg.prompt_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name, revision=cfg.model.revision
    )

    rows: List[Dict[str, Any]] = []
    for split, examples in prepared.items():
        for example in examples:
            messages, target = build_messages(example, instruction, 0)
            sft_example = build_worker_sft_example(
                tokenizer, messages, target
            )
            rows.append(
                {
                    "split": split,
                    "example_id": example.example_id,
                    "answerable": example.answerable,
                    "num_documents": len(example.documents),
                    "prompt_len": sft_example.prompt_len,
                    "target_len": int(sft_example.target_token_ids.shape[0]),
                    "total_len": int(sft_example.input_ids.shape[0]),
                    "over_max_input_tokens": (
                        int(sft_example.input_ids.shape[0])
                        > cfg.model.max_input_tokens
                    ),
                }
            )

    total_lens = sorted(row["total_len"] for row in rows)
    prompt_lens = sorted(row["prompt_len"] for row in rows)
    target_lens = sorted(row["target_len"] for row in rows)
    over_limit = [row for row in rows if row["over_max_input_tokens"]]

    report = {
        "tokenizer": {
            "name": cfg.model.name,
            "revision": cfg.model.revision,
            "vocab_size": tokenizer.vocab_size,
        },
        "max_input_tokens": cfg.model.max_input_tokens,
        "example_count": len(rows),
        "percentiles": {
            "total_len": _percentiles(total_lens),
            "prompt_len": _percentiles(prompt_lens),
            "target_len": _percentiles(target_lens),
        },
        "max_len_example": max(rows, key=lambda r: r["total_len"])
        if rows
        else None,
        "over_limit_count": len(over_limit),
        "over_limit_examples": over_limit,
        "per_split_counts": {
            split: sum(1 for row in rows if row["split"] == split)
            for split in prepared
        },
        "rows": rows,
    }
    path = cfg.prepared_data_dir() / "lengths_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"examples: {len(rows)}; total_len p50="
        f"{_percentiles(total_lens)['p50']} "
        f"p99={_percentiles(total_lens)['p99']} "
        f"max={_percentiles(total_lens)['max']}"
    )
    if over_limit:
        print(
            f"WARNING: {len(over_limit)} examples exceed "
            f"max_input_tokens={cfg.model.max_input_tokens}; the trainer "
            "treats this as a hard error. Inspect the report and raise "
            "the limit deliberately, never truncate silently."
        )
    return path


def _percentiles(sorted_values: List[int]) -> Dict[str, int]:
    if not sorted_values:
        return {"min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}

    def _pct(fraction: float) -> int:
        index = min(
            len(sorted_values) - 1,
            int(round(fraction * (len(sorted_values) - 1))),
        )
        return sorted_values[index]

    return {
        "min": sorted_values[0],
        "p50": _pct(0.5),
        "p90": _pct(0.9),
        "p99": _pct(0.99),
        "max": sorted_values[-1],
    }
