"""CLI override and run-count invariant tests."""

from __future__ import annotations

import pytest

from hotpot_mas.cli import _build_parser, _load_config
from hotpot_mas.config import ExperimentConfig


def _parse(*extra: str):
    return _build_parser().parse_args(
        ["run", "--config", "configs/base.yaml", *extra]
    )


def test_cli_override_destinations_are_applied():
    cfg = _load_config(
        _parse("--mode", "full", "--questions", "3", "--runs", "2")
    )
    assert cfg.num_questions == 3
    assert cfg.runs_per_question == 2
    assert cfg.run_seeds == [0, 1]


def test_seed_override_sets_run_count():
    cfg = _load_config(_parse("--seeds", "7,8,9"))
    assert cfg.runs_per_question == 3
    assert cfg.run_seeds == [7, 8, 9]


def test_protocol_fallback_pilot_mode_is_selectable():
    args = _build_parser().parse_args(
        [
            "run",
            "--config",
            "configs/protocol_fallback.yaml",
            "--mode",
            "pilot",
        ]
    )
    cfg = _load_config(args)
    assert cfg.num_questions == 20
    assert cfg.runs_per_question == 2
    assert cfg.run_seeds == [0, 1]


def test_mismatched_runs_and_seeds_are_rejected():
    with pytest.raises(ValueError, match="must equal len"):
        _load_config(_parse("--runs", "2", "--seeds", "7,8,9"))


def test_base_uses_transformers_engine():
    cfg = ExperimentConfig.from_yaml("configs/base.yaml")
    assert cfg.engine == "transformers"


def test_vllm_comparison_config_changes_only_backend_identity():
    base = ExperimentConfig.from_yaml("configs/base.yaml")
    vllm = ExperimentConfig.from_yaml("configs/vllm.yaml")
    assert vllm.engine == "vllm"
    assert vllm.experiment_version == "v5_vllm"
    assert vllm.model_name == base.model_name
    assert vllm.model_revision == base.model_revision
    assert vllm.tokenizer_revision == base.tokenizer_revision
    assert vllm.generation.to_dict() == base.generation.to_dict()
    assert vllm.prompt_dir == base.prompt_dir
    assert vllm.run_seeds == base.run_seeds


def test_invalid_engine_is_rejected(base_config):
    base_config.engine = "unknown"
    with pytest.raises(ValueError, match="engine must be"):
        base_config.validate()


def test_clarification_config_is_isolated_from_base():
    base = ExperimentConfig.from_yaml("configs/protocol_fallback.yaml")
    clarification = ExperimentConfig.from_yaml("configs/clarification.yaml")
    assert base.architecture == clarification.architecture == "mas"
    assert base.worker_clarification.enabled is False
    assert clarification.worker_clarification.enabled is True
    assert clarification.worker_clarification.max_per_worker == 1
    assert clarification.experiment_version == "v7_worker_clarification"
    assert clarification.model_name == base.model_name
    assert clarification.model_revision == base.model_revision
    assert clarification.tokenizer_revision == base.tokenizer_revision
    assert clarification.generation.to_dict() == base.generation.to_dict()
    assert clarification.run_seeds == base.run_seeds


def test_centralized_reader_config_uses_same_model_data_and_decoding():
    base = ExperimentConfig.from_yaml("configs/protocol_fallback.yaml")
    centralized = ExperimentConfig.from_yaml("configs/centralized_reader.yaml")
    assert centralized.architecture == "centralized_reader"
    assert centralized.worker_clarification.enabled is False
    assert centralized.model_name == base.model_name
    assert centralized.model_revision == base.model_revision
    assert centralized.tokenizer_revision == base.tokenizer_revision
    assert centralized.generation.to_dict() == base.generation.to_dict()
    assert centralized.manifest_path == base.manifest_path
    assert centralized.run_seeds == base.run_seeds


def test_diagnose_cli_accepts_three_run_files():
    args = _build_parser().parse_args(
        [
            "diagnose",
            "--baseline",
            "base.jsonl",
            "--clarification",
            "clarification.jsonl",
            "--centralized",
            "centralized.jsonl",
        ]
    )
    assert args.baseline == "base.jsonl"
    assert args.output_dir.endswith("reading_vs_communication")
