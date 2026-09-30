"""Deterministic run sharding and safe merge tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import pytest

from hotpot_mas.question_selection import build_manifest
from hotpot_mas.sharding import (
    load_questions_for_config,
    merge_shard_outputs,
    select_question_shard,
    shard_output_dir,
)


def _row(index: int) -> Dict[str, Any]:
    return {
        "id": f"q{index}",
        "question": f"Question {index}?",
        "answer": f"Answer {index}",
        "type": "bridge",
        "supporting_facts": [["Doc A", 0], ["Doc B", 0]],
        "context": [
            {"title": "Doc A", "sentences": [f"Alpha {index}."]},
            {"title": "Doc B", "sentences": [f"Beta {index}."]},
        ],
    }


def _config_with_manifest(base_config, tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    build_manifest(
        [_row(index) for index in range(5)],
        num_questions=5,
        selection_seed=0,
        manifest_path=manifest_path,
    )
    return replace(
        base_config,
        experiment_id="shard-test",
        experiment_version="v1",
        manifest_path=manifest_path,
        output_dir=tmp_path / "outputs",
        num_questions=5,
        runs_per_question=2,
        run_seeds=[0, 1],
    )


def _record(cfg, question: Dict[str, Any], seed: int, shard: Dict[str, int]):
    run_id = f"{question['question_id']}-seed-{seed}"
    return {
        "experiment_id": cfg.experiment_id,
        "experiment_version": cfg.experiment_version,
        "architecture": cfg.architecture,
        "run_id": run_id,
        "run_seed": seed,
        "question_id": question["question_id"],
        "question": question["question"],
        "gold_answer": question["answer"],
        "question_type": question["q_type"],
        "run_fingerprint": f"fingerprint-{run_id}",
        "shard": shard,
        "config": cfg.to_dict(),
        "f1": 1.0,
        "em": 1.0,
        "generated_tokens_alice": 1,
        "generated_tokens_bob": 1,
        "generated_tokens_celab": 1,
        "total_generated_tokens": 3,
        "input_tokens_alice": 2,
        "input_tokens_bob": 2,
        "input_tokens_celab": 2,
        "input_tokens_total": 6,
        "decision_steps": 1,
        "num_alice_queries": 1,
        "num_bob_queries": 1,
        "num_messages": 3,
        "num_clarification_requests": 0,
        "num_clarification_round_trips_completed": 0,
        "cap_reached": False,
        "generation_cap_reached": False,
        "natural_termination": True,
        "forced_final_calls": 0,
        "final_parse_fallback": False,
        "protocol_parse_fallback": False,
        "casefold_route_fallback": False,
        "terminal_final_precedence_fallback": False,
        "clarification_triggered": False,
        "clarification_protocol_violations": 0,
        "termination_reason": "direct_answer",
        "events": [],
    }


def _write_complete_shards(cfg, num_shards: int) -> None:
    questions = load_questions_for_config(cfg)
    for shard_index in range(num_shards):
        path = shard_output_dir(
            cfg, num_shards, shard_index
        ) / "runs.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "num_shards": num_shards,
            "shard_index": shard_index,
        }
        records = [
            _record(cfg, question, seed, metadata)
            for question in select_question_shard(
                questions, num_shards, shard_index
            )
            for seed in cfg.run_seeds
        ]
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )


def test_question_shards_are_balanced_disjoint_and_complete(
    base_config, tmp_path: Path
):
    cfg = _config_with_manifest(base_config, tmp_path)
    questions = load_questions_for_config(cfg)
    shards = [select_question_shard(questions, 4, index) for index in range(4)]
    assert [len(shard) for shard in shards] == [2, 1, 1, 1]
    ids = [question["question_id"] for shard in shards for question in shard]
    assert len(ids) == len(set(ids)) == 5
    assert set(ids) == {question["question_id"] for question in questions}


def test_merge_validates_completeness_and_writes_english_report(
    base_config, tmp_path: Path
):
    cfg = _config_with_manifest(base_config, tmp_path)
    _write_complete_shards(cfg, 4)
    summary = merge_shard_outputs(cfg, 4)
    assert summary["expected_runs"] == summary["merged_runs"] == 10
    assert summary["per_shard_run_counts"] == [4, 2, 2, 2]
    merged = Path(summary["merged_runs_path"])
    assert len(merged.read_text(encoding="utf-8").splitlines()) == 10
    report = Path(summary["report_paths"]["report"])
    assert report.is_file()
    assert "## Answer quality" in report.read_text(encoding="utf-8")


def test_merge_refuses_an_incomplete_shard(base_config, tmp_path: Path):
    cfg = _config_with_manifest(base_config, tmp_path)
    _write_complete_shards(cfg, 4)
    path = shard_output_dir(cfg, 4, 2) / "runs.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        merge_shard_outputs(cfg, 4)
    assert not (cfg.output_subdir() / "runs.jsonl").exists()
