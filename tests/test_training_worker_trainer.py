"""End-to-end Stage 1 trainer on FakePolicy/FakeSynthesizer (CPU-only).

Covers: shared-theta updates for A/B, exact G*G C calls per question,
cached old logprobs belonging to the rollout policy, metrics.jsonl
fields, seed reproducibility, attempt-bounded collection, checkpointing
and resume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

import hotpot_mas.training.data as training_data
from hotpot_mas.training.data import prepare_manifests
from hotpot_mas.training.fake_policy import FakePolicy, FakeSynthesizer
from hotpot_mas.training.worker_trainer import WorkerTrainer
from tests.training_test_utils import (
    MATRIX_BOTH,
    MATRIX_NONE,
    WRONG_ANSWER,
    make_rows,
    make_training_config,
    scripted_synthesizer,
)


def all_both_scripts(rows: List[Dict[str, Any]]) -> Dict[str, List[List[str]]]:
    return {f"Question text for {row['id']}?": MATRIX_BOTH for row in rows}


def make_prepared_trainer(
    tmp_path: Path,
    rows: List[Dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    scripts: Dict[str, List[List[str]]],
    *,
    policy: FakePolicy = None,
    resume: bool = False,
    **worker_overrides: Any,
) -> WorkerTrainer:
    monkeypatch.setattr(
        training_data,
        "load_hotpotqa_validation",
        lambda **kwargs: rows,
    )
    cfg = make_training_config(tmp_path, **worker_overrides)
    prepare_manifests(cfg)
    synth, _ = scripted_synthesizer(scripts, cfg.workers.G)
    trainer = WorkerTrainer(
        cfg,
        mode="smoke",
        resume=resume,
        policy_factory=lambda: policy or FakePolicy(),
        synthesizer_factory=lambda: synth,
    )
    return trainer


def read_metrics(stage_dir: Path) -> List[Dict[str, Any]]:
    path = stage_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestEndToEndUpdate:
    def test_ab_share_one_theta_and_only_theta_is_trained(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        rows = make_rows(8)
        policy = FakePolicy()
        theta0 = policy.logits.detach().clone()
        trainer = make_prepared_trainer(
            tmp_path,
            rows,
            monkeypatch,
            all_both_scripts(rows),
            policy=policy,
        )
        trainer.train()
        # Exactly one trainable parameter set: the shared theta. C is a
        # separate object with no parameters at all.
        params = trainer.optimizer.param_groups[0]["params"]
        assert len(params) == 1
        assert params[0] is policy.logits
        assert not torch.allclose(policy.logits.detach(), theta0)

    def test_c_called_g_squared_times_per_question(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        rows = make_rows(8)
        monkeypatch.setattr(
            training_data,
            "load_hotpotqa_validation",
            lambda **kwargs: rows,
        )
        cfg = make_training_config(tmp_path)
        prepare_manifests(cfg)
        synth = FakeSynthesizer(lambda q, a, b: WRONG_ANSWER)
        trainer = WorkerTrainer(
            cfg,
            mode="smoke",
            policy_factory=FakePolicy,
            synthesizer_factory=lambda: synth,
        )
        trainer.train()
        g = cfg.workers.G
        # Every answer misses the gold, so no question ever signals: the
        # collector consumes its whole attempt budget without any update
        # (and without validation evaluation), leaving exactly
        # attempts * G * G C calls.
        expected_total = cfg.workers.max_sampling_attempts * g * g
        assert len(synth.calls) == expected_total
        # Within one rollout C is called exactly G*G times, all for the
        # same question; consecutive rollouts cycle the train questions.
        cycle = [
            q.question for q in trainer.train_questions
        ] * cfg.workers.max_sampling_attempts
        for k in range(0, expected_total, g * g):
            chunk = synth.calls[k:k + g * g]
            assert len(chunk) == g * g
            assert {call[0] for call in chunk} == {cycle[k // (g * g)]}

    def test_cached_old_logprobs_belong_to_rollout_policy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        rows = make_rows(8)

        class RecordingPolicy(FakePolicy):
            def __init__(self):
                super().__init__()
                self.sample_theta: List[torch.Tensor] = []
                self.tf_theta: List[torch.Tensor] = []

            def sample_report(self, messages, generation_seed, decode):
                self.sample_theta.append(self.logits.detach().clone())
                return super().sample_report(messages, generation_seed, decode)

            def teacher_force(self, input_ids, completion_ids):
                self.tf_theta.append(self.logits.detach().clone())
                return super().teacher_force(input_ids, completion_ids)

        policy = RecordingPolicy()
        trainer = make_prepared_trainer(
            tmp_path,
            rows,
            monkeypatch,
            all_both_scripts(rows),
            policy=policy,
            num_policy_epochs=2,
        )
        trainer.train()
        theta0 = policy.sample_theta[0]
        # Both sides signal (MATRIX_BOTH): 2G = 4 kept reports. The first
        # epoch's teacher forcing (4 calls) must see the same theta that
        # sampled the rollout; the second epoch sees the stepped theta.
        assert len(policy.tf_theta) == 8
        for theta in policy.tf_theta[:4]:
            assert torch.allclose(theta, theta0)
        assert not torch.allclose(policy.tf_theta[4], theta0)

    def test_metrics_jsonl_has_all_required_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        rows = make_rows(8)
        trainer = make_prepared_trainer(
            tmp_path,
            rows,
            monkeypatch,
            all_both_scripts(rows),
        )
        trainer.train()
        records = read_metrics(trainer.stage_dir)
        assert len(records) == 1
        record = records[0]
        for key in [
            "run_id",
            "update",
            "mode",
            "attempted_questions",
            "signal_questions",
            "discarded_questions",
            "discard_rate",
            "signal_rate_a",
            "signal_rate_b",
            "signal_rate_both",
            "mean_reward",
            "mean_val_f1",
            "mean_val_em",
            "mean_a_tokens",
            "mean_b_tokens",
            "mean_c_tokens",
            "loss",
            "grad_norm",
            "approx_kl",
            "clip_fraction",
            "entropy",
            "checkpoint_path",
        ]:
            assert key in record, f"missing metrics field {key}"
        assert record["signal_questions"] == 1
        assert record["attempted_questions"] == 1
        assert Path(record["checkpoint_path"]).is_dir()

    def test_fixed_seed_reproduces_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        rows = make_rows(8)
        scripts = all_both_scripts(rows)
        trainer1 = make_prepared_trainer(
            tmp_path / "run1", rows, monkeypatch, scripts
        )
        trainer1.train()
        trainer2 = make_prepared_trainer(
            tmp_path / "run2", rows, monkeypatch, scripts
        )
        trainer2.train()
        records1 = read_metrics(trainer1.stage_dir)
        records2 = read_metrics(trainer2.stage_dir)
        assert len(records1) == len(records2) == 1
        for key in records1[0]:
            if key in ("checkpoint_path",):
                continue
            assert records1[0][key] == records2[0][key], f"diverged on {key}"

    def test_no_signal_questions_terminate_with_attempt_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        rows = make_rows(8)
        trainer = make_prepared_trainer(
            tmp_path,
            rows,
            monkeypatch,
            {f"Question text for {row['id']}?": MATRIX_NONE for row in rows},
            max_sampling_attempts=5,
        )
        outcome = trainer.train()
        assert outcome["updates_completed"] == 0
        assert read_metrics(trainer.stage_dir) == []

    def test_fail_on_insufficient_signal_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        rows = make_rows(8)
        trainer = make_prepared_trainer(
            tmp_path,
            rows,
            monkeypatch,
            {f"Question text for {row['id']}?": MATRIX_NONE for row in rows},
            max_sampling_attempts=5,
            fail_on_insufficient_signal=True,
        )
        with pytest.raises(RuntimeError, match="no signal"):
            trainer.train()


class TestCheckpointAndResume:
    def test_resume_continues_from_latest_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        rows = make_rows(8)
        scripts = all_both_scripts(rows)
        trainer1 = make_prepared_trainer(
            tmp_path, rows, monkeypatch, scripts, steps=1
        )
        trainer1.train()
        assert read_metrics(trainer1.stage_dir)[0]["update"] == 1

        trainer2 = make_prepared_trainer(
            tmp_path, rows, monkeypatch, scripts, steps=2, resume=True
        )
        trainer2.train()
        records = read_metrics(trainer2.stage_dir)
        assert [r["update"] for r in records] == [1, 2]
        assert trainer2.start_step == 1

    def test_checkpoint_dir_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        rows = make_rows(8)
        trainer = make_prepared_trainer(
            tmp_path, rows, monkeypatch, all_both_scripts(rows)
        )
        trainer.train()
        ckpt = trainer.stage_dir / "step_00001"
        assert ckpt.is_dir()
        assert (ckpt / "adapter").is_dir()
        assert (ckpt / "optimizer.pt").is_file()
        assert (ckpt / "state.json").is_file()
        state = json.loads((ckpt / "state.json").read_text(encoding="utf-8"))
        assert state["step"] == 1
        # The stored adapter reproduces the trained theta.
        fresh = FakePolicy()
        fresh.load_adapter(ckpt / "adapter")
        assert torch.allclose(
            fresh.logits.detach(), trainer.policy.logits.detach()
        )
