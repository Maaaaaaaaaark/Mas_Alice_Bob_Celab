"""Tests for streaming one-shot MAS failure attribution."""

from __future__ import annotations

import json
from pathlib import Path

from hotpot_mas.failure_analysis import write_failure_attribution_report
from hotpot_mas.logging_io import JsonlWriter


def _run(
    run_id: str,
    *,
    evidence_alice: str,
    evidence_bob: str,
    alice_report: str,
    bob_report: str,
    final: str | None,
    em: float,
    f1: float,
    termination: str = "direct_answer",
) -> dict:
    question_id, seed_text = run_id.rsplit("-seed-", 1)
    events = [
        {
            "event_type": "model_generation",
            "speaker": "alice",
            "raw_output": alice_report,
        },
        {
            "event_type": "model_generation",
            "speaker": "bob",
            "raw_output": bob_report,
        },
        {
            "event_type": "model_generation",
            "speaker": "celab",
            "raw_output": final,
        },
    ]
    return {
        "run_id": run_id,
        "question_id": question_id,
        "run_seed": int(seed_text),
        "config": {"run_seeds": [0]},
        "question": "Where is the answer?",
        "gold_answer": "Paris",
        "private_evidence_alice": evidence_alice,
        "private_evidence_bob": evidence_bob,
        "events": events,
        "final_answer": final,
        "termination_reason": termination,
        "f1": f1,
        "em": em,
        "total_generated_tokens": 10,
        "generation_cap_reached": False,
        "parse_error": None,
        "error": None,
    }


def test_failure_attribution_partitions_non_em_runs(tmp_path: Path):
    runs_path = tmp_path / "runs.jsonl"
    records = [
        _run(
            "correct-seed-0",
            evidence_alice="Paris is in France.",
            evidence_bob="Other evidence.",
            alice_report="The answer is Paris.",
            bob_report="Nothing relevant.",
            final="Paris",
            em=1.0,
            f1=1.0,
        ),
        _run(
            "overcomplete-seed-0",
            evidence_alice="Paris is in France.",
            evidence_bob="Other evidence.",
            alice_report="The answer is Paris.",
            bob_report="Nothing relevant.",
            final="Paris, France",
            em=0.0,
            f1=2 / 3,
        ),
        _run(
            "synthesis-seed-0",
            evidence_alice="Paris is in France.",
            evidence_bob="Other evidence.",
            alice_report="The answer is Paris.",
            bob_report="Nothing relevant.",
            final="London",
            em=0.0,
            f1=0.0,
        ),
        _run(
            "extraction-seed-0",
            evidence_alice="Paris is in France.",
            evidence_bob="Other evidence.",
            alice_report="No answer found.",
            bob_report="Nothing relevant.",
            final="London",
            em=0.0,
            f1=0.0,
        ),
        _run(
            "nonlexical-seed-0",
            evidence_alice="The capital is referred to indirectly.",
            evidence_bob="Other evidence.",
            alice_report="No answer found.",
            bob_report="Nothing relevant.",
            final="London",
            em=0.0,
            f1=0.0,
        ),
        _run(
            "system-seed-0",
            evidence_alice="Paris is in France.",
            evidence_bob="Other evidence.",
            alice_report="The answer is Paris.",
            bob_report="Nothing relevant.",
            final=None,
            em=0.0,
            f1=0.0,
            termination="error",
        ),
    ]
    writer = JsonlWriter(runs_path)
    for record in records:
        writer.append(record)

    written = write_failure_attribution_report(runs_path, tmp_path / "analysis")
    summary = json.loads(
        written["failure_attribution_summary"].read_text(encoding="utf-8")
    )
    assert summary["total_runs"] == 6
    assert summary["failed_runs"] == 5
    categories = summary["primary_failure_categories"]
    assert categories["system_or_protocol_failure"]["count"] == 1
    assert categories["final_answer_overcomplete"]["count"] == 1
    assert categories["synthesis_selection_failure"]["count"] == 1
    assert categories["worker_extraction_failure"]["count"] == 1
    assert (
        categories["answer_not_lexically_explicit_in_evidence"]["count"] == 1
    )
    assert summary["question_level"]["questions_with_all_expected_seeds"] == 6
    report = written["failure_attribution_report"].read_text(encoding="utf-8")
    assert "One-Shot MAS Failure Attribution" in report
    assert "Heuristic observational attribution" in report
    assert "Where is the gold span in private evidence?" in report


def test_failure_attribution_reports_worker_location_and_partial_lines(
    tmp_path: Path,
):
    runs_path = tmp_path / "runs.jsonl"
    record = _run(
        "both-seed-0",
        evidence_alice="Paris appears here.",
        evidence_bob="Paris also appears here.",
        alice_report="Paris",
        bob_report="Paris",
        final="Paris",
        em=1.0,
        f1=1.0,
    )
    runs_path.write_text(json.dumps(record) + "\n{partial", encoding="utf-8")
    written = write_failure_attribution_report(runs_path, tmp_path / "analysis")
    summary = json.loads(
        written["failure_attribution_summary"].read_text(encoding="utf-8")
    )
    assert summary["malformed_lines_ignored"] == 1
    assert summary["gold_span_location_in_private_evidence"]["both"]["count"] == 1
    assert summary["gold_span_location_in_worker_reports"]["both"]["count"] == 1
    assert (
        summary["gold_span_location_in_worker_reports"]["both"]
        ["exact_match_rate"]
        == 1.0
    )
