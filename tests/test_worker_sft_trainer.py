"""End-to-end trainer tests for Stage 1a, using the fake model on CPU.

A fake-model run is a functional test ONLY, not a training result: the
trainer records ``real_model=False`` and the tests assert it is reported
honestly. Real training runs on the cluster with the real model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import pytest
import torch

from hotpot_mas.training.worker_sft.trainer import (
    WorkerSFTTrainer,
    _plan_groups,
    run_worker_sft_training,
)

from worker_sft_test_utils import (
    build_cpu_config,
    make_doc,
    make_evidence,
    make_example,
    make_fake_model_for,
    write_prepared_data,
    write_trace_support_files,
)


def _train_set() -> List[Any]:
    examples = []
    for i in range(4):
        doc = make_doc("d0", f"The mine opened in 18{i}0. It is famous.")
        examples.append(
            make_example(
                f"train-pos-{i}",
                f"When did mine {i} open?",
                [doc],
                answer=f"18{i}0",
                evidence=[make_evidence("d0", f"The mine opened in 18{i}0.")],
            )
        )
    doc = make_doc("d0", "The village has a church and a school.")
    examples.append(
        make_example(
            "train-neg-0",
            "When was the village founded?",
            [doc],
            answer="1900",
            answerable=False,
            negative_reason="official unanswerable",
        )
    )
    return examples


def _val_set() -> List[Any]:
    doc = make_doc("d0", "The bridge was built in 1905.")
    return [
        make_example(
            "val-0",
            "When was the bridge built?",
            [doc],
            answer="1905",
            evidence=[make_evidence("d0", "The bridge was built in 1905.")],
        )
    ]


def _probe_set() -> List[Any]:
    d0 = make_doc("d0", "Arthur's Magazine began in 1844.", "Arthur")
    d1 = make_doc("d1", "In May 1846 it was merged into Godey's.", "Godey")
    return [
        make_example(
            "probe-0",
            "When was Arthur's Magazine merged into Godey's?",
            [d0, d1],
            answer="May 1846",
            evidence=[
                make_evidence("d0", "Arthur's Magazine began in 1844."),
                make_evidence("d1", "In May 1846 it was merged into Godey's."),
            ],
            probe=True,
        )
    ]


def test_plan_groups_splits_with_correct_partial_tail():
    assert _plan_groups(10, 3) == [3, 3, 3, 1]
    assert _plan_groups(6, 3) == [3, 3]
    assert _plan_groups(0, 3) == []
    assert _plan_groups(5, 1) == [1, 1, 1, 1, 1]


def test_smoke_training_runs_grad_accumulation_with_partial_group(tmp_path):
    cfg = build_cpu_config(tmp_path, grad_accum_steps=3)
    train = _train_set()
    write_prepared_data(cfg, train, val=_val_set(), probes=_probe_set())
    fake = make_fake_model_for(train + _val_set() + _probe_set())
    base_before = fake.base_param_hash()
    lora_before = fake.trainable_snapshot()

    outcome = run_worker_sft_training(
        cfg, "smoke", model_factory=lambda: fake
    )

    # 5 examples, group sizes [3, 2]: the final PARTIAL group keeps
    # per-example weighting and still produces exactly 2 optimizer steps.
    assert outcome["optimizer_steps"] == 2
    assert outcome["base_weights_unchanged"] is True
    stage_dir = Path(outcome["stage_dir"])
    metrics = [
        json.loads(line)
        for line in (stage_dir / "metrics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [m["microsteps"] for m in metrics] == [3, 2]
    for m in metrics:
        assert abs(m["update_loss"] - sum(m["per_example_losses"]) / m["microsteps"]) < 1e-6
    # LoRA moved, base did not.
    assert fake.trainable_delta_norm(lora_before) > 0.0
    assert fake.base_param_hash() == base_before
    # Final checkpoint exists with adapter + state.
    final = Path(outcome["final_checkpoint"])
    assert (final / "adapter.pt").is_file()
    state_path = final.parent / "state.json"
    assert state_path.is_file()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["step"] == 2
    # Fixed pre/post generations are written with all checks.
    outputs = json.loads(
        (stage_dir / "fixed_example_outputs.json").read_text(
            encoding="utf-8"
        )
    )
    assert {item["example_id"] for item in outputs["inputs"]} >= {
        "train-pos-0", "probe-0"
    }
    for item in outputs["inputs"]:
        assert set(item["pre"].keys()) >= {"text", "num_new_tokens", "checks"}
        assert set(item["post"].keys()) >= {"text", "num_new_tokens", "checks"}


def test_trace_mode_builds_trace_and_worked_example(tmp_path):
    cfg = build_cpu_config(tmp_path, grad_accum_steps=1)
    train = _train_set()
    write_prepared_data(cfg, train, val=_val_set(), probes=_probe_set())
    write_trace_support_files(cfg)
    fake = make_fake_model_for(train + _val_set() + _probe_set())

    outcome = run_worker_sft_training(
        cfg, "trace", model_factory=lambda: fake
    )

    assert outcome["optimizer_steps"] == len(train)
    stage_dir = Path(outcome["stage_dir"])
    trace = json.loads(
        (stage_dir / "trace.json").read_text(encoding="utf-8")
    )
    for section in (
        "A_run_identity", "B_raw_data", "C_data_conversion",
        "D_dedup_and_selection", "E_inputs_targets", "F_tokenization",
        "G_loss", "H_parameter_update", "I_generation", "J_limits",
    ):
        assert section in trace, f"missing trace section {section}"
    # Real-model flag is recorded honestly for the fake run.
    assert trace["A_run_identity"]["model"]["real_model"] is False
    # Per-token NLLs come from the ACTUAL forward; the loss arithmetic
    # must reproduce the recorded update loss.
    for step in trace["H_parameter_update"]["per_step"]:
        recomputed = sum(step["per_example_losses"]) / step["microsteps"]
        assert abs(recomputed - step["update_loss"]) < 1e-6
    md_path = Path(outcome["worked_example"])
    assert md_path.is_file()
    md = md_path.read_text(encoding="utf-8")
    for header in ("## A. Run identity", "## J. Limits and next steps"):
        assert header in md
    # The worked example never fabricates values: what is rendered is
    # exactly what the trace recorded.
    assert "update loss" in md
    # Checkpoint per step in trace mode, plus the final checkpoint.
    checkpoints = sorted(stage_dir.glob("step_*"))
    assert len(checkpoints) == len(train) + 1


def test_rendered_worked_example_is_deterministic(tmp_path):
    cfg = build_cpu_config(tmp_path, grad_accum_steps=1)
    train = _train_set()
    write_prepared_data(cfg, train, val=_val_set(), probes=_probe_set())
    write_trace_support_files(cfg)
    fake = make_fake_model_for(train + _val_set() + _probe_set())
    outcome = run_worker_sft_training(
        cfg, "trace", model_factory=lambda: fake
    )
    first = Path(outcome["worked_example"]).read_text(encoding="utf-8")
    from hotpot_mas.training.worker_sft.trace import (
        render_worker_sft_worked_example,
    )

    again = Path(outcome["stage_dir"]) / "rendered_again.md"
    render_worker_sft_worked_example(
        Path(outcome["stage_dir"]) / "trace.json", again
    )
    assert again.read_text(encoding="utf-8") == first


def test_trainer_refuses_to_overwrite_existing_run(tmp_path):
    cfg = build_cpu_config(tmp_path, grad_accum_steps=1)
    train = _train_set()
    write_prepared_data(cfg, train, val=_val_set())
    fake = make_fake_model_for(train + _val_set())
    run_worker_sft_training(cfg, "smoke", model_factory=lambda: fake)
    with pytest.raises(FileExistsError):
        run_worker_sft_training(cfg, "smoke", model_factory=lambda: fake)


def test_resume_continues_from_latest_checkpoint(tmp_path):
    cfg = build_cpu_config(tmp_path, grad_accum_steps=1)
    train = _train_set()
    write_prepared_data(cfg, train, val=_val_set())
    # Trace mode loads the two report files, so they must exist for the
    # first run as well as the resumed one.
    write_trace_support_files(cfg)
    fake = make_fake_model_for(train + _val_set())
    first = run_worker_sft_training(
        cfg, "trace", model_factory=lambda: fake
    )
    assert first["optimizer_steps"] == len(train)

    resumed_fake = make_fake_model_for(train + _val_set())
    resumed = run_worker_sft_training(
        cfg, "trace", resume=True, model_factory=lambda: resumed_fake
    )
    # Resume starts after the completed steps: the remaining group plan
    # is empty, so zero NEW optimizer steps are taken.
    assert resumed["optimizer_steps"] == 0
    # The resumed model loaded the checkpoint adapter: its LoRA matches
    # the trained fake's LoRA.
    for (name, param), (resumed_name, resumed_param) in zip(
        fake.trainable_snapshot().items(),
        resumed_fake.trainable_snapshot().items(),
    ):
        assert name == resumed_name
        assert torch.allclose(param, resumed_param, atol=1e-6)


def test_resume_requires_existing_checkpoints(tmp_path):
    cfg = build_cpu_config(tmp_path, grad_accum_steps=1)
    write_prepared_data(cfg, _train_set())
    fake = make_fake_model_for(_train_set())
    trainer = WorkerSFTTrainer(
        cfg, "smoke", resume=True, model_factory=lambda: fake
    )
    with pytest.raises(FileNotFoundError, match="no step checkpoints"):
        trainer.train()


def test_trainer_requires_prepared_data(tmp_path):
    cfg = build_cpu_config(tmp_path, grad_accum_steps=1)
    fake = make_fake_model_for([])
    with pytest.raises(FileNotFoundError, match="prepare-worker-sft-data"):
        run_worker_sft_training(
            cfg, "smoke", model_factory=lambda: fake
        )
