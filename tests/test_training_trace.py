"""Trace mode: trace.json completeness, worked_example.md auto-generation,
deterministic re-rendering, and trace/metrics content checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import hotpot_mas.training.data as training_data
from hotpot_mas.training.data import prepare_manifests
from hotpot_mas.training.fake_policy import FakePolicy
from hotpot_mas.training.trace import (
    REQUIRED_KEYS,
    load_trace_json,
    render_worked_example,
    write_trace_json,
)
from hotpot_mas.training.worker_trainer import WorkerTrainer
from tests.training_test_utils import (
    MATRIX_BOTH,
    make_rows,
    make_training_config,
    scripted_synthesizer,
)


def run_trace_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> WorkerTrainer:
    """One trace-mode update: 1 train question, G=2, 2 policy epochs."""
    rows = make_rows(8)
    monkeypatch.setattr(
        training_data,
        "load_hotpotqa_validation",
        lambda **kwargs: rows,
    )
    cfg = make_training_config(
        tmp_path,
        train_num=1,
        steps=1,
        questions_per_update=1,
        num_policy_epochs=2,
        max_sampling_attempts=30,
    )
    prepare_manifests(cfg)
    scripts = {f"Question text for {row['id']}?": MATRIX_BOTH for row in rows}
    synth, _ = scripted_synthesizer(scripts, cfg.workers.G)
    trainer = WorkerTrainer(
        cfg,
        mode="trace",
        policy_factory=FakePolicy,
        synthesizer_factory=lambda: synth,
    )
    trainer.train()
    return trainer


class TestTraceModeRun:
    def test_trace_json_has_all_required_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        trainer = run_trace_trainer(tmp_path, monkeypatch)
        trace_path = trainer.stage_dir / "trace.json"
        assert trace_path.is_file()
        data = load_trace_json(trace_path)  # raises when incomplete
        assert set(REQUIRED_KEYS) <= set(data)
        assert data["dataset"]["question_id"]
        assert len(data["reward_matrix"]) == 2
        assert data["loss"]["per_report"]
        # Two policy epochs on one rollout batch: the trace keeps the last
        # epoch's teacher-forced values, and the diagnostic mirrors them.
        assert data["diagnostics"]["num_policy_epochs"] == 2
        assert data["kept"]["sides"] == ["A", "B"]

    def test_trace_json_contains_gold_only_in_reward_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        trainer = run_trace_trainer(tmp_path, monkeypatch)
        data = load_trace_json(trainer.stage_dir / "trace.json")
        gold = data["question"]["gold"]
        # The gold answer appears in the question record (as y*) but never
        # inside any worker prompt or C input.
        for key in ("worker_a_messages", "worker_b_messages"):
            for message in data["prompts"][key]:
                assert gold not in message["content"]
        for row in data["pairs"]:
            for pair in row:
                for message in pair["messages"]:
                    assert gold not in message["content"]

    def test_worked_example_generated_and_rerender_is_deterministic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        trainer = run_trace_trainer(tmp_path, monkeypatch)
        md_path = trainer.stage_dir / "worked_example.md"
        assert md_path.is_file()
        first = md_path.read_text(encoding="utf-8")
        render_worked_example(
            trainer.stage_dir / "trace.json", tmp_path / "rerendered.md"
        )
        second = (tmp_path / "rerendered.md").read_text(encoding="utf-8")
        assert second == first

    def test_worked_example_numbers_match_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        trainer = run_trace_trainer(tmp_path, monkeypatch)
        data = load_trace_json(trainer.stage_dir / "trace.json")
        md = (trainer.stage_dir / "worked_example.md").read_text(
            encoding="utf-8"
        )
        # Spot-check: the recorded values must appear in the markdown
        # formatted from the trace (nothing hand-fabricated).
        assert f"{data['loss']['J']:.4f}" in md
        assert f"{data['loss']['L']:.4f}" in md
        assert f"{data['reward_matrix'][0][0]:.4f}" in md
        assert f"{data['diagnostics']['grad_norm']:.4f}" in md
        assert f"{data['diagnostics']['approx_kl']:.4f}" in md
        assert f"{data['diagnostics']['clip_fraction']:.4f}" in md

    def test_metrics_line_written_with_val_note(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        trainer = run_trace_trainer(tmp_path, monkeypatch)
        lines: List[Dict[str, Any]] = [
            json.loads(line)
            for line in (trainer.stage_dir / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert len(lines) == 1
        record = lines[0]
        assert record["mode"] == "trace"
        assert record["val_note"] is not None
        assert record["mean_val_f1"] is None
        assert record["num_policy_epochs"] == 2

    def test_checkpoint_and_environment_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        trainer = run_trace_trainer(tmp_path, monkeypatch)
        data = load_trace_json(trainer.stage_dir / "trace.json")
        assert data["checkpoint"]["saved"] is True
        assert Path(data["checkpoint"]["path"]).is_dir()
        assert data["environment"]
        assert "python_version" in data["environment"]


class TestTraceValidation:
    def test_write_trace_json_rejects_missing_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        trainer = run_trace_trainer(tmp_path, monkeypatch)
        data = load_trace_json(trainer.stage_dir / "trace.json")
        del data["loss"]
        with pytest.raises(ValueError, match="missing required fields"):
            write_trace_json(tmp_path / "broken.json", data)

    def test_load_trace_json_rejects_incomplete_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        trainer = run_trace_trainer(tmp_path, monkeypatch)
        broken = tmp_path / "broken.json"
        data = json.loads(
            (trainer.stage_dir / "trace.json").read_text(encoding="utf-8")
        )
        del data["reward_matrix"]
        broken.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="not a complete trace"):
            load_trace_json(broken)
