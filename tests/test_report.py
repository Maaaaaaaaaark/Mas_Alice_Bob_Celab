"""End-to-end report generation regression tests."""

from __future__ import annotations

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

