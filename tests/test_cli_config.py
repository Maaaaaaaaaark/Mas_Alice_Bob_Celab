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


def test_intermediate_controls_change_only_declared_mechanisms():
    base = ExperimentConfig.from_yaml("configs/protocol_fallback.yaml")
    shared = ExperimentConfig.from_yaml("configs/shared_question.yaml")
    one_shot = ExperimentConfig.from_yaml("configs/one_shot_gather.yaml")
    direct = ExperimentConfig.from_yaml(
        "configs/one_shot_direct_answer.yaml"
    )

    assert shared.architecture == "mas"
    assert shared.share_question_with_workers is True
    assert shared.worker_clarification.enabled is False
    assert shared.experiment_version == "v8_shared_question"

    assert one_shot.architecture == "one_shot_gather"
    assert one_shot.share_question_with_workers is True
    assert one_shot.worker_clarification.enabled is False
    assert one_shot.experiment_version == "v9_one_shot_gather"

    assert direct.architecture == "one_shot_direct_answer"
    assert direct.share_question_with_workers is True
    assert direct.worker_clarification.enabled is False
    assert direct.experiment_version == "v10_one_shot_direct_answer"

    for condition in (shared, one_shot, direct):
        assert condition.model_name == base.model_name
        assert condition.model_revision == base.model_revision
        assert condition.tokenizer_revision == base.tokenizer_revision
        assert condition.generation.to_dict() == base.generation.to_dict()
        assert condition.manifest_path == base.manifest_path
        assert condition.run_seeds == base.run_seeds


def test_one_shot_requires_worker_question_visibility(base_config):
    base_config.architecture = "one_shot_gather"
    base_config.share_question_with_workers = False
    with pytest.raises(ValueError, match="share_question_with_workers"):
        base_config.validate()


def test_distractor_experiment_configs_are_paired_and_auditable():
    mas = ExperimentConfig.from_yaml(
        "configs/one_shot_direct_answer_distractor.yaml"
    )
    single = ExperimentConfig.from_yaml(
        "configs/single_agent_distractor.yaml"
    )
    assert mas.evidence_partition == single.evidence_partition == (
        "balanced_distractor"
    )
    assert mas.partition_seed == single.partition_seed == 0
    assert mas.manifest_path == single.manifest_path
    assert mas.run_seeds == single.run_seeds
    assert mas.generation.to_dict() == single.generation.to_dict()
    assert mas.model_instance_mode == "independent"
    assert mas.agent_devices == {
        "alice": "cuda:0",
        "bob": "cuda:0",
        "celab": "cuda:0",
    }
    assert single.model_instance_mode == "shared"
    assert single.architecture == "centralized_reader"
    assert mas.modes["full"].num_questions == 7345
    assert single.modes["full"].num_questions == 7345


def test_independent_instances_may_share_one_device(base_config):
    base_config.model_instance_mode = "independent"
    base_config.agent_devices = {
        "alice": "cuda:0",
        "bob": "cuda:0",
        "celab": "cuda:0",
    }
    base_config.validate()


def test_independent_instances_require_all_agent_device_keys(base_config):
    base_config.model_instance_mode = "independent"
    base_config.agent_devices = {"alice": "cuda:0", "bob": "cuda:0"}
    with pytest.raises(ValueError, match="exactly alice, bob, and celab"):
        base_config.validate()


def test_diagnose_cli_accepts_all_five_run_files():
    args = _build_parser().parse_args(
        [
            "diagnose",
            "--baseline",
            "base.jsonl",
            "--clarification",
            "clarification.jsonl",
            "--centralized",
            "centralized.jsonl",
            "--shared-question",
            "shared.jsonl",
            "--one-shot",
            "one-shot.jsonl",
        ]
    )
    assert args.baseline == "base.jsonl"
    assert args.shared_question == "shared.jsonl"
    assert args.one_shot == "one-shot.jsonl"
    assert args.output_dir.endswith("reading_vs_communication")


def test_run_cli_accepts_question_shard_arguments():
    args = _build_parser().parse_args(
        [
            "run",
            "--config",
            "configs/one_shot_direct_answer_distractor.yaml",
            "--mode",
            "full",
            "--num-shards",
            "4",
            "--shard-index",
            "2",
        ]
    )
    assert args.num_shards == 4
    assert args.shard_index == 2


def test_merge_shards_cli_defaults_to_full_mode():
    args = _build_parser().parse_args(
        [
            "merge-shards",
            "--config",
            "configs/single_agent_distractor.yaml",
            "--num-shards",
            "4",
        ]
    )
    assert args.mode == "full"
    assert args.num_shards == 4
