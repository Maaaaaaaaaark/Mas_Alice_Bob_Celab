"""Trace recorder and worked-example renderer tests for Stage 1a.
CPU-only; no model checkpoint, no GPU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from hotpot_mas.training.worker_sft.fakes import FakeWorkerTokenizer
from hotpot_mas.training.worker_sft.masking import build_worker_sft_example
from hotpot_mas.training.worker_sft.trace import (
    REQUIRED_SECTIONS,
    WorkerSFTTraceRecorder,
    render_worker_sft_worked_example,
)

from worker_sft_test_utils import (
    build_cpu_config,
    make_doc,
    make_evidence,
    make_example,
    minimal_worker_sft_trace,
)

MESSAGES = [
    {"role": "system", "content": "You are a search worker."},
    {"role": "user", "content": "Question: When?\n\nDocuments:\n\nD0: Title\nText."},
]
TARGET = 'D0: "Text."'

REQUIRED_HEADERS = [
    "## A. Run identity",
    "## B. Raw data",
    "## C. Data conversion",
    "## D. Dedup and sample selection",
    "## E. Actual worker inputs and targets",
    "## F. Tokenization and the loss mask",
    "## G. SFT loss",
    "## H. Parameter updates",
    "## I. Before/after generation",
    "## J. Limits and next steps",
]


def _tokenizer() -> FakeWorkerTokenizer:
    return FakeWorkerTokenizer(
        corpus=[
            "You are a search worker.",
            "When?",
            "Text.",
            "Documents:",
            "D0:",
            '"Text."',
            "NOT",
            "FOUND",
        ]
    )


def _example() -> Any:
    doc = make_doc("d0", "The tower opened in 1889.")
    return make_example(
        "e1",
        "When did the tower open?",
        [doc],
        answer="1889",
        evidence=[make_evidence("d0", "The tower opened in 1889.")],
    )


def _sft_example() -> Any:
    return build_worker_sft_example(_tokenizer(), MESSAGES, TARGET)


# --------------------------------------------------------------------------
# Section builders (unit level)
# --------------------------------------------------------------------------

def test_section_a_reads_seeds_and_revision_only_from_snapshot(tmp_path):
    cfg = build_cpu_config(tmp_path)
    section = WorkerSFTTraceRecorder._section_a(
        start_time="2026-10-12T00:00:00+00:00",
        run_id="r1",
        config_snapshot=cfg.to_dict(),
        model_info={
            "model_name": "fake/base-model",
            "chat_template_sha256": "abc",
            "real_model": False,
            "fresh_adapter": True,
            "loaded_previous_adapter": False,
        },
        env_info={"platform": "test"},
    )
    assert section["run_id"] == "r1"
    assert section["model"]["real_model"] is False
    # No revision pinned in the fixture config -> honest placeholder.
    assert section["model"]["revision"] == (
        "resolved at load time (pin and re-run to record)"
    )
    assert section["seeds"]["seed"] == cfg.seed
    assert section["seeds"]["probe_seed"] == cfg.decode.probe_seed


def test_section_f_builds_token_tables_with_mask_and_eot(tmp_path):
    sft_example = _sft_example()
    section = WorkerSFTTraceRecorder._section_f(
        sft_examples={"e1": sft_example},
        train_examples=[_example()],
    )
    data = section["examples"]["e1"]
    counts = data["counts"]
    assert counts["prompt_tokens"] == sft_example.prompt_len
    assert counts["total_tokens"] == int(sft_example.input_ids.shape[0])
    assert counts["target_tokens"] == int(
        sft_example.target_token_ids.shape[0]
    )
    table = data["token_table"]
    assert len(table) == counts["total_tokens"]
    # Prompt positions are masked (no loss), target positions supervised.
    assert all(
        not row["contributes_to_loss"]
        for row in table[: sft_example.prompt_len]
    )
    assert all(
        row["contributes_to_loss"]
        for row in table[sft_example.prompt_len :]
    )
    assert table[sft_example.prompt_len - 1]["label"] == -100
    eot = data["eot_supervision"]
    assert eot["eos_trained"] is True
    assert eot["first_target_trained"] is True
    assert eot["eos_position"] == counts["total_tokens"] - 1
    # Causal alignment: first target token predicted from last prompt pos.
    assert str(sft_example.prompt_len - 1) in data["causal_alignment"]


def test_section_g_computes_example_loss_from_recorded_nlls(tmp_path):
    sft_example = _sft_example()
    nll_row = torch.full((len(sft_example.input_ids) - 1,), 0.3)
    section = WorkerSFTTraceRecorder._section_g(
        sft_examples={"e1": sft_example},
        per_example_nll_cache={"e1": nll_row},
        train_examples=[_example()],
        per_step_records=[
            {
                "step": 0,
                "update_loss": 1.5,
                "microsteps": 2,
                "per_example_losses": [1.0, 2.0],
            }
        ],
    )
    assert "plain English" in section["teacher_forcing"]
    row = section["per_example"][0]
    assert len(row["supervised_tokens"]) == len(sft_example.target_token_ids)
    for token in row["supervised_tokens"]:
        assert token["nll"] == pytest.approx(0.3)
        assert token["log_prob"] == pytest.approx(-0.3)
        assert token["contributes_to_loss"] is True
    # The example loss is the mean of its supervised token NLLs.
    assert row["example_loss"] == pytest.approx(0.3)
    assert "= 0.300000" in row["arithmetic"]
    # The update loss arithmetic is spelled out from the real records.
    update = section["update_losses"][0]
    assert "1.000000 + 2.000000 / 2 = 1.500000" in update["arithmetic"]


def test_section_h_lists_checkpoints_and_frozen_base_evidence(tmp_path):
    section = WorkerSFTTraceRecorder._section_h(
        per_step_records=[
            {"checkpoint": str(Path("step_00000") / "adapter")},
            {"checkpoint": None},
        ],
        config_snapshot={"optimizer": {"name": "AdamW"}},
        model_info={"frozen_audit": {"trainable_names": ["lora"]}},
        base_hash_before="aaa",
        base_hash_after="aaa",
        base_unchanged=True,
    )
    assert section["checkpoints"] == [str(Path("step_00000") / "adapter")]
    assert section["frozen_base_check"]["unchanged"] is True
    assert "sha256" in section["frozen_base_check"]["method"]


def test_section_j_names_confirmed_not_established_and_deferred():
    section = WorkerSFTTraceRecorder._section_j()
    assert set(section) == {"confirmed", "not_established", "deferred"}
    assert all(section.values())


# --------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------

def test_renderer_writes_all_section_headers(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(minimal_worker_sft_trace()), encoding="utf-8"
    )
    out_path = tmp_path / "worked_example_worker_sft.md"
    render_worker_sft_worked_example(trace_path, out_path)
    md = out_path.read_text(encoding="utf-8")
    assert md.startswith("# Worked example: Stage 1a worker SFT (trace)")
    for header in REQUIRED_HEADERS:
        assert header in md
    # The renderer never hides the fixture flag.
    assert "Real model (not a fixture): False" in md


def test_renderer_is_deterministic(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(minimal_worker_sft_trace()), encoding="utf-8"
    )
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    render_worker_sft_worked_example(trace_path, first)
    render_worker_sft_worked_example(trace_path, second)
    assert first.read_text(encoding="utf-8") == second.read_text(
        encoding="utf-8"
    )


def test_renderer_refuses_traces_with_missing_sections(tmp_path):
    trace = minimal_worker_sft_trace()
    del trace["G_loss"]
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="missing section"):
        render_worker_sft_worked_example(
            trace_path, tmp_path / "out.md"
        )


def test_required_sections_cover_a_through_j():
    assert REQUIRED_SECTIONS == [
        "A_run_identity",
        "B_raw_data",
        "C_data_conversion",
        "D_dedup_and_selection",
        "E_inputs_targets",
        "F_tokenization",
        "G_loss",
        "H_parameter_update",
        "I_generation",
        "J_limits",
    ]
