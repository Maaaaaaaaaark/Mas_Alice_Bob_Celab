"""End-to-end report generation regression tests."""

from __future__ import annotations

import json
from pathlib import Path

from hotpot_mas.logging_io import JsonlWriter
from hotpot_mas.report import write_report


def test_report_uses_gold_answer_field(run_with_mock, tmp_path: Path):
    record, _ = run_with_mock()
    runs_path = tmp_path / "runs.jsonl"
    JsonlWriter(runs_path).append(record)
    paths = write_report(runs_path, tmp_path)
    assert all(path.is_file() for path in paths.values())
    per_question = paths["summary_per_question"].read_text(encoding="utf-8")
    assert '"gold_answer": "The Connector Bridge"' in per_question


def test_report_counts_unclosed_final_fallback(run_with_mock, tmp_path: Path):
    record, _ = run_with_mock(
        scripts={"celab": ["Celab: <FINAL>The Connector Bridge"]}
    )
    runs_path = tmp_path / "runs.jsonl"
    JsonlWriter(runs_path).append(record)
    paths = write_report(runs_path, tmp_path)
    summary = json.loads(
        paths["summary_run_level"].read_text(encoding="utf-8")
    )
    rate = summary["rates"]["unclosed_final_fallback_rate"]
    assert rate == {"rate": 1.0, "count": 1, "n": 1}
    report = paths["report"].read_text(encoding="utf-8")
    assert "Unclosed Final Fallback Rate" in report


def test_report_counts_protocol_fallback_types(run_with_mock, tmp_path: Path):
    record, _ = run_with_mock(
        scripts={
            "celab": [
                "Celab: <TO>ALICE</TO> x <FINAL>The Connector Bridge</FINAL>"
            ]
        }
    )
    runs_path = tmp_path / "runs.jsonl"
    JsonlWriter(runs_path).append(record)
    paths = write_report(runs_path, tmp_path)
    summary = json.loads(
        paths["summary_run_level"].read_text(encoding="utf-8")
    )
    assert summary["rates"]["protocol_parse_fallback_rate"]["count"] == 1
    assert (
        summary["rates"]["terminal_final_precedence_fallback_rate"]["count"]
        == 1
    )
    assert summary["rates"]["casefold_route_fallback_rate"]["count"] == 0
