"""Paired reading-vs-communication diagnostic report tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hotpot_mas.diagnostics import write_diagnostic_report
from hotpot_mas.logging_io import JsonlWriter


def _record(
    question_id: str,
    seed: int,
    f1: float,
    em: float,
    tokens: int,
) -> dict:
    return {
        "run_id": f"{question_id}-seed-{seed}",
        "question_id": question_id,
        "run_seed": seed,
        "model_name": "google/gemma-3-1b-it",
        "model_revision": "revision",
        "tokenizer_revision": "revision",
        "f1": f1,
        "em": em,
        "total_generated_tokens": tokens,
        "input_tokens_total": 100,
        "termination_reason": "direct_answer",
    }


def test_diagnostic_report_is_english_and_paired(tmp_path: Path):
    paths = {
        name: tmp_path / name / "runs.jsonl"
        for name in ("base", "clarification", "centralized")
    }
    for question in ("q1", "q2"):
        for seed in (0, 1):
            JsonlWriter(paths["base"]).append(
                _record(question, seed, 0.1, 0.0, 300)
            )
            JsonlWriter(paths["clarification"]).append(
                _record(question, seed, 0.2, 0.0, 340)
            )
            JsonlWriter(paths["centralized"]).append(
                _record(question, seed, 0.8, 1.0, 30)
            )

    written = write_diagnostic_report(
        paths["base"],
        paths["clarification"],
        paths["centralized"],
        tmp_path / "diagnostic",
    )
    payload = json.loads(
        written["diagnostic_summary"].read_text(encoding="utf-8")
    )
    comparison = payload["paired_comparisons_against_baseline"][
        "centralized_reader"
    ]["f1"]
    assert comparison["matched_runs"] == 4
    assert comparison["matched_questions"] == 2
    assert comparison["mean_delta"] == pytest.approx(0.7)
    assert "communication coordination" in payload["diagnosis"]
    report = written["diagnostic_report"].read_text(encoding="utf-8")
    assert "Reading-vs-Communication Diagnostic" in report
    assert "question-bootstrap 95% CI" in report


def test_five_condition_report_contains_staged_comparisons(tmp_path: Path):
    paths = {
        name: tmp_path / name / "runs.jsonl"
        for name in (
            "base",
            "clarification",
            "shared",
            "one_shot",
            "centralized",
        )
    }
    scores = {
        "base": (0.1, 300),
        "clarification": (0.05, 800),
        "shared": (0.2, 250),
        "one_shot": (0.4, 100),
        "centralized": (0.7, 20),
    }
    for question in ("q1", "q2"):
        for seed in (0, 1):
            for name, (f1, tokens) in scores.items():
                JsonlWriter(paths[name]).append(
                    _record(question, seed, f1, float(f1 == 1.0), tokens)
                )

    written = write_diagnostic_report(
        paths["base"],
        paths["clarification"],
        paths["centralized"],
        tmp_path / "diagnostic",
        shared_question_path=paths["shared"],
        one_shot_path=paths["one_shot"],
    )
    payload = json.loads(
        written["diagnostic_summary"].read_text(encoding="utf-8")
    )
    stages = payload["staged_paired_comparisons"]
    assert stages["question_visibility"]["f1"]["mean_delta"] == pytest.approx(
        0.1
    )
    assert stages["fixed_schedule"]["f1"]["mean_delta"] == pytest.approx(0.2)
    assert stages["centralization"]["f1"]["mean_delta"] == pytest.approx(0.3)

    report = written["diagnostic_report"].read_text(encoding="utf-8")
    assert "Shared-question MAS" in report
    assert "One-shot gather MAS" in report
    assert "Staged mechanism comparisons" in report
    assert "effect of worker question visibility" in report
    # Every non-baseline condition still has a complete baseline comparison.
    assert report.count("over 4 matched runs and 2 questions") == 12
