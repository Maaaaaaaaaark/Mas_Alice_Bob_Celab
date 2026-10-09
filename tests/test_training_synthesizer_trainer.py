"""CPU-only tests for Stage 2 (synthesizer_trainer.py).

Everything runs offline: a fake Stage 1 run directory supplies the worker
checkpoint, ``FakePolicy`` stands in for the frozen workers, and
``FakeTrainableSynthesizer`` is C with a trainable LoRA logits parameter.
The exact report-cache, SFT-loss, checkpoint and metrics code paths of the
GPU trainer run unchanged here.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pytest
import torch

from hotpot_mas.training.config import DecodeConfig
from hotpot_mas.training.eval import QuestionEval, SplitEvalResult
from hotpot_mas.training.fake_policy import (
    FakePolicy,
    FakeTrainableSynthesizer,
)
from hotpot_mas.training.synthesizer_trainer import (
    SynthesizerTrainer,
    check_worker_adapter_compatibility,
    resolve_worker_checkpoint,
)
from tests.training_test_utils import (
    build_stage2_manifests,
    make_fake_worker_run,
    make_synthesizer_training_config,
)


# ----------------------------------------------------------------------
# best-checkpoint resolution
# ----------------------------------------------------------------------

class TestResolveWorkerCheckpoint:
    def test_best_step_not_latest(self, tmp_path) -> None:
        run_dir, expected = make_fake_worker_run(
            tmp_path, {1: 0.3, 2: 0.9, 3: 0.5, 4: 0.7}
        )
        ckpt = resolve_worker_checkpoint(run_dir)
        assert ckpt.best_step == 2  # not the latest step 4
        assert ckpt.best_val_f1 == 0.9
        assert ckpt.identity.startswith("step_00002-")
        assert ckpt.resolution == "metrics_jsonl"
        assert ckpt.adapter_dir == run_dir / "step_00002" / "adapter"
        assert ckpt.adapter_dir.is_dir()

    def test_direct_adapter_dir(self, tmp_path) -> None:
        run_dir, _ = make_fake_worker_run(
            tmp_path, {1: 0.3, 2: 0.9, 3: 0.5}
        )
        adapter_dir = run_dir / "step_00002" / "adapter"
        ckpt = resolve_worker_checkpoint(adapter_dir)
        assert ckpt.adapter_dir == adapter_dir
        assert ckpt.best_step == 2
        assert ckpt.identity.startswith("step_00002-")
        assert ckpt.resolution == "adapter_dir"

    def test_missing_val_f1_raises(self, tmp_path) -> None:
        run_dir, _ = make_fake_worker_run(
            tmp_path, {}, steps=3  # every step has mean_val_f1: null
        )
        with pytest.raises(RuntimeError, match="no validation F1"):
            resolve_worker_checkpoint(run_dir)

    def test_metrics_state_conflict_raises(self, tmp_path) -> None:
        run_dir, _ = make_fake_worker_run(tmp_path, {1: 0.3, 2: 0.9})
        # metrics.jsonl resolves step 2; corrupt the final state.json.
        final_state = (
            run_dir / "step_00002" / "state.json"
        )
        state = json.loads(final_state.read_text(encoding="utf-8"))
        state["best_step"] = 1
        final_state.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(RuntimeError, match="conflict"):
            resolve_worker_checkpoint(run_dir)

    def test_missing_adapter_raises(self, tmp_path) -> None:
        run_dir, _ = make_fake_worker_run(tmp_path, {1: 0.3, 2: 0.9})
        import shutil

        shutil.rmtree(run_dir / "step_00002" / "adapter")
        with pytest.raises(FileNotFoundError, match="adapter"):
            resolve_worker_checkpoint(run_dir)

    def test_no_step_dirs_raises(self, tmp_path) -> None:
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="no step_\\*"):
            resolve_worker_checkpoint(run_dir)

    def test_state_only_fallback(self, tmp_path) -> None:
        run_dir, expected = make_fake_worker_run(
            tmp_path, {1: 0.3, 2: 0.9}
        )
        (run_dir / "metrics.jsonl").unlink()
        ckpt = resolve_worker_checkpoint(run_dir)
        assert ckpt.best_step == expected["best_step"] == 2
        assert ckpt.best_val_f1 == expected["best_val_f1"] == 0.9
        assert ckpt.resolution == "state_json"

    def test_checkpoint_identity_changes_with_adapter_content(
        self, tmp_path
    ) -> None:
        run_dir, _ = make_fake_worker_run(tmp_path, {1: 0.3, 2: 0.9})
        first = resolve_worker_checkpoint(run_dir).identity
        adapter_file = run_dir / "step_00002" / "adapter" / "weights.bin"
        adapter_file.write_bytes(b"different trained weights")
        second = resolve_worker_checkpoint(run_dir).identity
        assert first != second

    def test_nonexistent_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_worker_checkpoint(tmp_path / "nope")

    def test_incompatible_adapter_raises(self, tmp_path) -> None:
        run_dir, _ = make_fake_worker_run(tmp_path, {1: 0.5, 2: 0.9})
        adapter_dir = run_dir / "step_00002" / "adapter"
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "other/model",
                    "r": 16,
                    "lora_alpha": 32,
                }
            ),
            encoding="utf-8",
        )
        cfg = make_synthesizer_training_config(tmp_path, run_dir)
        ckpt = resolve_worker_checkpoint(run_dir)
        with pytest.raises(RuntimeError, match="incompatible"):
            check_worker_adapter_compatibility(ckpt, cfg)

    def test_compatible_adapter_passes(self, tmp_path) -> None:
        run_dir, _ = make_fake_worker_run(tmp_path, {1: 0.5, 2: 0.9})
        adapter_dir = run_dir / "step_00002" / "adapter"
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "fake/base-model",
                    "r": 16,
                    "lora_alpha": 32,
                }
            ),
            encoding="utf-8",
        )
        cfg = make_synthesizer_training_config(tmp_path, run_dir)
        ckpt = resolve_worker_checkpoint(run_dir)
        compat = check_worker_adapter_compatibility(ckpt, cfg)
        assert compat is not None and compat["checked"] is True

    def test_missing_adapter_config_skips_check(self, tmp_path) -> None:
        run_dir, _ = make_fake_worker_run(tmp_path, {1: 0.5, 2: 0.9})
        cfg = make_synthesizer_training_config(tmp_path, run_dir)
        ckpt = resolve_worker_checkpoint(run_dir)
        assert check_worker_adapter_compatibility(ckpt, cfg) is None


# ----------------------------------------------------------------------
# end-to-end trainer
# ----------------------------------------------------------------------

@pytest.fixture
def stage2(tmp_path) -> Dict[str, Any]:
    run_dir, expected = make_fake_worker_run(tmp_path, {1: 0.5, 2: 0.9})
    paths = build_stage2_manifests(tmp_path)
    cfg = make_synthesizer_training_config(
        tmp_path,
        run_dir,
        worker_eval_decode=DecodeConfig(
            do_sample=False, temperature=0.0, top_p=1.0, max_new_tokens=4
        ),
    )
    return {"tmp_path": tmp_path, "run_dir": run_dir, "cfg": cfg,
            "expected": expected}


def make_trainer(cfg) -> Tuple[SynthesizerTrainer, FakePolicy,
                                FakeTrainableSynthesizer]:
    policy = FakePolicy()
    synthesizer = FakeTrainableSynthesizer()
    trainer = SynthesizerTrainer(
        cfg,
        mode="smoke",
        policy_factory=lambda: policy,
        synthesizer_factory=lambda: synthesizer,
    )
    return trainer, policy, synthesizer


class TestSynthesizerTrainerEndToEnd:
    def test_worker_frozen_and_released(self, stage2) -> None:
        trainer, policy, _ = make_trainer(stage2["cfg"])
        # Frozen: no trainable parameters left on the worker policy.
        assert policy.trainable_parameters() == []
        assert not policy.logits.requires_grad
        # Released: the trainer holds no worker reference any more.
        assert not hasattr(trainer, "worker_policy")
        before = policy.logits.detach().clone()
        trainer.train()
        assert torch.equal(policy.logits, before)

    def test_only_c_lora_trainable(self, stage2) -> None:
        trainer, _, synthesizer = make_trainer(stage2["cfg"])
        assert synthesizer.trainable_parameters() == [synthesizer.lora_logits]
        assert not synthesizer.base_logits.requires_grad
        base_before = synthesizer.base_logits.clone()
        lora_before = synthesizer.lora_logits.detach().clone()
        trainer.train()
        assert torch.equal(synthesizer.base_logits, base_before)
        assert not torch.equal(synthesizer.lora_logits.detach(), lora_before)

    def test_step0_baseline_and_epoch0_metrics(self, stage2) -> None:
        trainer, _, _ = make_trainer(stage2["cfg"])
        stage_dir = stage2["cfg"].stage_dir()
        trainer.train()
        baseline_path = stage_dir / "step0_baseline.json"
        assert baseline_path.is_file()
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert baseline["epoch"] == 0
        for key in (
            "val_with_reports",
            "val_empty_reports",
        ):
            assert key in baseline
            assert baseline[key]["mean_f1"] == 0.0  # fresh C0 answers empty
            assert baseline[key]["mean_em"] == 0.0
        metrics = [
            json.loads(line)
            for line in (stage_dir / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        epoch0 = [m for m in metrics if m["epoch"] == 0]
        assert len(epoch0) == 1
        assert epoch0[0]["train_loss"] is None
        assert epoch0[0]["mean_val_f1"] == 0.0
        assert epoch0[0]["empty_val_f1"] == 0.0

    def test_c0_and_c_phi_share_identical_reports(self, stage2) -> None:
        trainer, policy, synthesizer = make_trainer(stage2["cfg"])
        trainer.train()
        # C0 covers val + val-empty at step 0 (4 calls), then 2 epochs of
        # val + val-empty (8 calls).  Only after validation selection, the
        # final C0 and C_phi test comparisons make 8 calls.
        calls = synthesizer.answer_calls
        assert len(calls) == 20
        # C0's test calls (with reports) == C_phi's test calls.
        assert calls[12:14] == calls[16:18]
        # C0's empty-test calls == C_phi's empty-test calls.
        assert calls[14:16] == calls[18:20]
        # The worker policy decoded exactly once per (question, side):
        # 4 train + 2 val + 2 test questions, A and B each.
        assert policy._calls.count("decode") == 16
        # The empty-control calls really use empty reports.
        for question, a_report, b_report in calls[14:16]:
            assert a_report == "" and b_report == ""

    def test_validation_best_checkpoint_restored(self, stage2) -> None:
        trainer, _, synthesizer = make_trainer(stage2["cfg"])
        stage_dir = stage2["cfg"].stage_dir()

        # Script the training and validation curves: epoch 1 improves to
        # F1 1.0, epoch 2 falls back to 0.0, so the epoch-1 checkpoint
        # must be restored at the end, not the last epoch's.
        def fake_train_epoch(epoch: int) -> Tuple[float, float]:
            synthesizer.lora_logits.data.fill_(float(epoch))
            return 1.0, 2.0

        val_counter = {"n": 0}
        val_f1_script = [0.0, 1.0, 0.0]  # step0, epoch1, epoch2

        def fake_evaluate_split(
            split: str, empty_control: bool = False
        ) -> SplitEvalResult:
            if split == "val" and not empty_control:
                f1 = val_f1_script[min(val_counter["n"], 2)]
                val_counter["n"] += 1
            else:
                f1 = 0.0
            return SplitEvalResult(
                split=split,
                questions=[
                    QuestionEval(
                        question_id=f"{split}-{i}",
                        f1=f1,
                        em=f1,
                        a_generated_tokens=0,
                        b_generated_tokens=0,
                        c_generated_tokens=3,
                    )
                    for i in range(2)
                ],
            )

        trainer._train_epoch = fake_train_epoch
        trainer._evaluate_split = fake_evaluate_split
        trainer.train()

        # The best checkpoint (epoch 1) was restored, not epoch 2.
        assert torch.all(
            synthesizer.lora_logits == 1.0
        ), "validation-best C must be restored, never the last epoch"
        best_state = json.loads(
            (stage_dir / "best_checkpoint" / "state.json").read_text(
                encoding="utf-8"
            )
        )
        assert best_state["epoch"] == 1
        assert best_state["best_epoch"] == 1
        assert best_state["best_val_f1"] == 1.0
        # Per-epoch checkpoints exist with the expected layout.
        for epoch in (1, 2):
            epoch_dir = stage_dir / f"epoch_{epoch:03d}"
            assert (epoch_dir / "adapter" / "fake_adapter.pt").is_file()
            assert (epoch_dir / "state.json").is_file()
        # The best adapter equals the epoch-1 adapter byte for byte.
        assert (stage_dir / "best_checkpoint" / "adapter" / "fake_adapter.pt"
                ).read_bytes() == (
            stage_dir / "epoch_001" / "adapter" / "fake_adapter.pt"
        ).read_bytes()

    def test_final_comparison_and_report(self, stage2) -> None:
        trainer, _, _ = make_trainer(stage2["cfg"])
        stage_dir = stage2["cfg"].stage_dir()
        trainer.train()
        comparison_path = stage_dir / "final_comparison.json"
        assert comparison_path.is_file()
        comparison = json.loads(
            comparison_path.read_text(encoding="utf-8")
        )
        for key in (
            "c0_with_reports",
            "c0_empty_reports",
            "c_phi_with_reports",
            "c_phi_empty_reports",
        ):
            assert key in comparison
            assert "mean_f1" in comparison[key]
            assert "mean_em" in comparison[key]
            assert "mean_c_tokens" in comparison[key]
            assert "parsed" in comparison[key]
        assert comparison["test_report_cache"]["num_questions"] == 2
        assert "identity" in comparison["test_report_cache"]
        assert set(comparison["test_ab_tokens"]) == {
            "mean_a_tokens", "mean_b_tokens", "mean_ab_tokens"
        }
        # Empty-report control recorded for C0 and C_phi.
        assert comparison["c0_empty_reports"]["mean_f1"] == 0.0
        assert comparison["c_phi_empty_reports"]["mean_f1"] == 0.0
        report_path = stage_dir / "report.md"
        assert report_path.is_file()
        report = report_path.read_text(encoding="utf-8")
        assert "C0 + reports" in report
        assert "C_phi + reports" in report
        assert "C0 + empty reports" in report
        assert "C_phi + empty reports" in report

    def test_non_finite_loss_raises(self, stage2) -> None:
        trainer, _, synthesizer = make_trainer(stage2["cfg"])
        vocab = len(synthesizer.tokenizer.vocab)

        def nan_logits(input_ids, attention_mask):
            return torch.full(
                (input_ids.shape[0], input_ids.shape[1], vocab),
                float("nan"),
            )

        synthesizer.sft_logits = nan_logits
        with pytest.raises(FloatingPointError, match="non-finite"):
            trainer.train()

    def test_fresh_run_refuses_existing_artifacts(self, stage2) -> None:
        stage_dir = stage2["cfg"].stage_dir()
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "metrics.jsonl").write_text(
            '{"run_id": "epoch-00000", "epoch": 0}\n', encoding="utf-8"
        )
        with pytest.raises(RuntimeError, match="already contains artifacts"):
            make_trainer(stage2["cfg"])

    def test_provenance_and_cache_metadata(self, stage2) -> None:
        trainer, _, _ = make_trainer(stage2["cfg"])
        stage_dir = stage2["cfg"].stage_dir()
        provenance_path = stage_dir / "worker_checkpoint_provenance.json"
        assert provenance_path.is_file()
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        assert provenance["best_step"] == stage2["expected"]["best_step"]
        assert provenance["best_val_f1"] == stage2["expected"]["best_val_f1"]
        assert provenance["resolution"] == "metrics_jsonl"
        assert "step_00002" in provenance["adapter_dir"]
        snapshot = json.loads(
            (stage_dir / "config_snapshot.json").read_text(encoding="utf-8")
        )
        assert snapshot["config"]["worker_checkpoint"] == str(
            stage2["run_dir"]
        )
        metadata = json.loads(
            (stage_dir / "report_cache_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        for split, count in (("train", 4), ("val", 2), ("test", 2)):
            entry = metadata["splits"][split]
            assert entry["num_questions"] == count
            assert entry["cache_identity"]
            assert (stage_dir / "report_cache" /
                    f"{entry['cache_identity']}.json").is_file()
        trainer.train()

    def test_trainer_returns_summary(self, stage2) -> None:
        trainer, _, _ = make_trainer(stage2["cfg"])
        summary = trainer.train()
        assert summary["best_epoch"] == 1
        assert summary["best_val_f1"] == 0.0
        assert summary["test_f1_c0"] == 0.0
        assert summary["test_f1_c_phi"] == 0.0

    def test_trace_writes_stage2_worked_example(self, stage2) -> None:
        trainer, _, _ = make_trainer(stage2["cfg"])
        trainer.mode = "trace"
        summary = trainer.train()
        path = stage2["cfg"].stage_dir() / "worked_example.md"
        assert summary["worked_example_path"] == str(path)
        text = path.read_text(encoding="utf-8")
        assert "Stage 2 worked example" in text
        assert "Alice frozen-worker report" in text
        assert "SFT target and loss mask" in text
        assert "C0 + reports" in text
        assert "C_phi + reports" in text
