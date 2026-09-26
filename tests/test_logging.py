"""Logging tests (spec sec. 15, 18): record schema, controller events,
token-sum consistency, resumable JSONL writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from hotpot_mas.logging_io import JsonlWriter, load_runs

REQUIRED_KEYS = {
    "experiment_id", "experiment_version", "dataset", "dataset_split",
    "run_id", "question_id", "run_index", "run_seed", "sample_selection_seed",
    "model_name", "model_revision", "tokenizer_revision", "generation_config",
    "question", "gold_answer", "question_type", "supporting_titles",
    "private_evidence_alice", "private_evidence_bob",
    "alice_private_context", "bob_private_context", "dataset_metadata",
    "supporting_facts", "partition_metadata",
    "prompt_version", "prompt_hashes", "config", "environment", "engine_info",
    "events", "decision_steps", "cap_reached", "decision_cap_reached",
    "generation_cap_reached", "forced_final_calls", "natural_termination",
    "termination_reason", "final_raw_output", "final_answer",
    "final_answer_extracted", "parse_status", "parse_error",
    "f1", "em", "answer_f1", "answer_em",
    "num_alice_queries", "num_bob_queries", "num_total_queries",
    "num_alice_responses", "num_bob_responses", "num_messages",
    "input_tokens_alice", "input_tokens_bob", "input_tokens_celab",
    "alice_input_tokens", "bob_input_tokens", "celab_input_tokens",
    "total_input_tokens",
    "input_tokens_total", "generated_tokens_alice", "generated_tokens_bob",
    "generated_tokens_celab", "total_generated_tokens", "total_model_tokens",
    "alice_generated_tokens", "bob_generated_tokens", "celab_generated_tokens",
    "generation_cap_agents", "generation_cap_events",
    "duration_seconds", "error",
}

EVENT_KEYS = {
    "event_index", "event_type", "decision_step", "speaker", "recipient",
    "message_id", "raw_output", "parsed_action", "parsed_body",
    "parse_status", "parse_error", "input_tokens", "generated_tokens",
    "generation_seed", "finish_reason", "generation_cap_reached", "forced_final",
    "controller_subtype", "visible_history_message_ids",
}


def test_run_record_schema_is_complete(run_with_mock):
    record, engine = run_with_mock()
    missing = REQUIRED_KEYS - set(record)
    assert not missing, f"missing run-level keys: {sorted(missing)}"
    for event in record["events"]:
        missing_event = EVENT_KEYS - set(event)
        assert not missing_event, f"missing event keys: {sorted(missing_event)}"
    # Spec sec. 15.3: every model_generation event records what the model saw.
    for event in record["events"]:
        if event["event_type"] == "model_generation":
            assert isinstance(event["visible_history_message_ids"], list)
            assert isinstance(event["message_id"], str)
            assert event["message_id"]
            assert isinstance(event["generation_seed"], int)
    # Controller events carry zero tokens and a subtype.
    for event in record["events"]:
        if event["event_type"] == "controller":
            assert event["generated_tokens"] == 0
            assert event["controller_subtype"] is not None


def test_controller_event_ids_are_visible_in_celab_history(run_with_mock):
    record, engine = run_with_mock()
    task_event = record["events"][0]
    assert task_event["controller_subtype"] == "initial_task"
    # The initial task message id appears in Celab's first call input,
    # because the question is delivered as Celab's first user turn.
    first_celab = engine.calls_by_speaker["celab"][0]
    first_celab_input = "\n".join(
        m["content"] for m in first_celab.messages
    )
    assert record["question"] in first_celab_input


def test_prompt_version_and_hashes_recorded(run_with_mock, prompts):
    record, engine = run_with_mock()
    assert record["prompt_version"] == prompts.version
    assert record["prompt_version"].startswith("v1-")
    assert set(record["prompt_hashes"]) == {"alice", "bob", "celab", "forced_final"}
    for name, digest in record["prompt_hashes"].items():
        assert digest == prompts.hashes[name]
        assert len(digest) == 64


def test_resolved_config_recorded(run_with_mock, base_config):
    record, engine = run_with_mock()
    cfg = record["config"]
    assert cfg["experiment_id"] == base_config.experiment_id
    assert cfg["generation"]["temperature"] == 0.6
    assert cfg["generation"]["top_p"] == 0.95
    assert cfg["generation"]["do_sample"] is True
    assert cfg["generation"]["max_new_tokens"] == 2048
    assert cfg["max_decision_steps"] == 20


def test_token_sums_consistent(run_with_mock):
    record, engine = run_with_mock()
    event_input = sum(e["input_tokens"] for e in record["events"])
    event_generated = sum(
        e["generated_tokens"] for e in record["events"]
        if e["event_type"] == "model_generation"
    )
    assert record["input_tokens_total"] == event_input
    assert record["total_generated_tokens"] == event_generated
    assert record["total_model_tokens"] == (
        record["input_tokens_total"] + record["total_generated_tokens"]
    )


def _make_record(run_id: str) -> Dict[str, Any]:
    return {"run_id": run_id, "f1": 0.5, "events": []}


class TestJsonlWriter:
    def test_append_and_reload(self, tmp_path: Path):
        path = tmp_path / "runs.jsonl"
        writer = JsonlWriter(path)
        writer.append(_make_record("a"))
        writer.append(_make_record("b"))
        runs = load_runs(path)
        assert [r["run_id"] for r in runs] == ["a", "b"]

    def test_resume_skips_existing_ids(self, tmp_path: Path):
        path = tmp_path / "runs.jsonl"
        writer = JsonlWriter(path)
        writer.append(_make_record("a"))
        writer.append(_make_record("b"))

        # Reopen on the same file: existing ids are known, appending "a"
        # again would be caught by the caller via contains().
        writer2 = JsonlWriter(path)
        assert writer2.contains("a") and writer2.contains("b")
        assert not writer2.contains("c")
        writer2.append(_make_record("c"))
        assert [r["run_id"] for r in load_runs(path)] == ["a", "b", "c"]

    def test_partial_trailing_line_is_ignored(self, tmp_path: Path):
        path = tmp_path / "runs.jsonl"
        path.write_text(
            json.dumps(_make_record("a")) + "\n" + '{"run_id": "trunca',
            encoding="utf-8",
        )
        writer = JsonlWriter(path)
        assert writer.contains("a")
        assert not writer.contains("trunca")
        runs = load_runs(path)
        assert len(runs) == 1
        writer.append(_make_record("b"))
        assert [r["run_id"] for r in load_runs(path)] == ["a", "b"]

    def test_missing_file_starts_empty(self, tmp_path: Path):
        writer = JsonlWriter(tmp_path / "nope.jsonl")
        assert not writer.contains("anything")
