"""Training config loading, validation, and mode overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from hotpot_mas.training.config import (
    DataConfig,
    SplitConfig,
    SFTConfig,
    SynthesizerTrainingConfig,
    TrainingConfig,
    WorkerTrainingConfig,
    deep_merge,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def base_yaml() -> Dict[str, Any]:
    return {
        "experiment_id": "test",
        "experiment_version": "v1",
        "seed": 0,
        "output_dir": "outputs/training",
        "manifest_dir": "outputs/training_manifests/test",
        "prompt_dir": "prompts_training",
        "data": {
            "dataset": "hotpot_qa",
            "dataset_config": "distractor",
            "train": {"split": "train", "num_questions": 4, "selection_seed": 0},
            "val": {"split": "validation", "num_questions": 2, "selection_seed": 1},
            "test": {"split": "validation", "num_questions": 2, "selection_seed": 2},
            "partition_seed": 0,
            "val_test_split_seed": 0,
        },
        "model": {
            "name": "google/gemma-3-1b-it",
            "dtype": "float16",
            "device": "cuda",
            "max_input_length": 30000,
        },
        "workers": {
            "G": 2,
            "questions_per_update": 2,
            "steps": 2,
        },
    }


class TestTrainingConfig:
    def test_load_and_defaults(self, tmp_path: Path):
        raw = base_yaml()
        raw["prompt_dir"] = str(REPO_ROOT / "prompts_training")
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        cfg = TrainingConfig.from_yaml(path)
        assert cfg.workers.G == 2
        # TeX-exact default: a single policy epoch per rollout batch.
        assert cfg.workers.num_policy_epochs == 1
        assert cfg.workers.clip_epsilon == 0.2
        assert cfg.workers.delta == 0.05
        assert cfg.model.dtype == "float16"
        assert cfg.prompt_dir.is_dir()

    def test_validation_rejects_bad_values(self, tmp_path: Path):
        for key, value in [
            ("G", 1),
            ("num_policy_epochs", 0),
            ("clip_epsilon", 0.0),
            ("delta", -0.1),
            ("learning_rate", 0.0),
            ("steps", 0),
            ("minibatch_size", 0),
            ("max_sampling_attempts", 0),
        ]:
            raw = base_yaml()
            raw["prompt_dir"] = str(REPO_ROOT / "prompts_training")
            raw["workers"][key] = value
            path = tmp_path / f"bad_{key}.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with pytest.raises(ValueError):
                TrainingConfig.from_yaml(path)

    def test_validation_rejects_invalid_sampling_distribution(
        self, tmp_path: Path
    ):
        for override in (
            {"top_p": 0.0},
            {"top_p": 1.1},
            {"do_sample": True, "temperature": 0.0},
        ):
            raw = base_yaml()
            raw["prompt_dir"] = str(REPO_ROOT / "prompts_training")
            raw["workers"]["rollout"] = {
                "max_new_tokens": 8,
                **override,
            }
            path = tmp_path / f"bad_decode_{len(list(tmp_path.iterdir()))}.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with pytest.raises(ValueError):
                TrainingConfig.from_yaml(path)

    def test_grpo_rollout_rejects_truncated_probability_support(
        self, tmp_path: Path
    ):
        raw = base_yaml()
        raw["prompt_dir"] = str(REPO_ROOT / "prompts_training")
        raw["workers"]["rollout"] = {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": 8,
        }
        path = tmp_path / "truncated_support.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="probability support"):
            TrainingConfig.from_yaml(path)

    def test_unknown_mode_rejected(self, tmp_path: Path):
        raw = base_yaml()
        raw["prompt_dir"] = str(REPO_ROOT / "prompts_training")
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValueError):
            TrainingConfig.from_yaml(path, mode="nope")

    def test_mode_deep_merge(self, tmp_path: Path):
        raw = base_yaml()
        raw["prompt_dir"] = str(REPO_ROOT / "prompts_training")
        raw["modes"] = {
            "trace": {
                "experiment_version": "trace",
                "workers": {
                    "G": 2,
                    "num_policy_epochs": 2,
                    "questions_per_update": 1,
                    "steps": 1,
                },
            }
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        cfg = TrainingConfig.from_yaml(path, mode="trace")
        assert cfg.experiment_version == "trace"
        # Merged override wins...
        assert cfg.workers.steps == 1
        assert cfg.workers.num_policy_epochs == 2
        # ...while untouched fields keep their base values.
        assert cfg.workers.G == 2
        assert cfg.workers.questions_per_update == 1
        assert cfg.data.train.split == "train"
        assert cfg.seed == 0

    def test_deep_merge_replaces_lists(self):
        merged = deep_merge(
            {"a": [1, 2], "b": {"x": 1}},
            {"a": [3], "b": {"y": 2}},
        )
        assert merged == {"a": [3], "b": {"x": 1, "y": 2}}

    def test_direct_construction_and_validation(self, tmp_path: Path):
        cfg = TrainingConfig(
            experiment_id="t",
            experiment_version="v1",
            seed=0,
            output_dir=tmp_path / "outputs",
            manifest_dir=tmp_path / "manifests",
            prompt_dir=REPO_ROOT / "prompts_training",
            data=DataConfig(
                train=SplitConfig("validation", 2, 0),
                val=SplitConfig("validation", 2, 1),
                test=SplitConfig("validation", 2, 2),
            ),
            workers=WorkerTrainingConfig(G=2, steps=1),
        )
        cfg.validate()  # must not raise
        assert cfg.stage_dir() == tmp_path / "outputs" / "t" / "v1"


class TestSynthesizerConfig:
    def test_worker_checkpoint_required(self, tmp_path: Path):
        raw = base_yaml()
        raw["prompt_dir"] = str(REPO_ROOT / "prompts_training")
        raw["experiment_id"] = "sft"
        raw["sft"] = {"num_epochs": 2}
        path = tmp_path / "sft.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="worker_checkpoint"):
            SynthesizerTrainingConfig.from_yaml(path)

    def test_sft_defaults(self):
        sft = SFTConfig()
        assert sft.num_epochs == 3
        assert sft.learning_rate == 1e-4
        assert sft.synthesizer_decode.do_sample is False

    def _write_sft_yaml(
        self, tmp_path: Path, extra: Dict[str, Any] = None,
        name: str = "sft.yaml",
    ) -> Path:
        raw = base_yaml()
        raw["prompt_dir"] = str(REPO_ROOT / "prompts_training")
        raw["experiment_id"] = "sft"
        raw["sft"] = {"num_epochs": 2}
        checkpoint = tmp_path / "worker_run"
        checkpoint.mkdir(exist_ok=True)
        raw["worker_checkpoint"] = str(checkpoint)
        for key, value in (extra or {}).items():
            raw[key] = value
        path = tmp_path / name
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return path

    def test_loads_with_worker_checkpoint_dir(self, tmp_path: Path):
        path = self._write_sft_yaml(tmp_path)
        cfg = SynthesizerTrainingConfig.from_yaml(path)
        assert Path(cfg.worker_checkpoint).is_dir()
        assert Path(cfg.worker_checkpoint).is_absolute()

    def test_worker_checkpoint_must_exist(self, tmp_path: Path):
        path = self._write_sft_yaml(tmp_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["worker_checkpoint"] = str(tmp_path / "missing_run")
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="does not exist"):
            SynthesizerTrainingConfig.from_yaml(path)

    def test_deterministic_decodes_enforced(self, tmp_path: Path):
        for field in ("worker_eval_decode", "synthesizer_decode"):
            path = self._write_sft_yaml(
                tmp_path,
                extra={
                    "sft": {
                        "num_epochs": 2,
                        field: {
                            "do_sample": True,
                            "temperature": 0.6,
                            "top_p": 1.0,
                            "max_new_tokens": 64,
                        },
                    }
                },
                name=f"bad_{field}.yaml",
            )
            with pytest.raises(ValueError, match="deterministic"):
                SynthesizerTrainingConfig.from_yaml(path)

    def test_bad_sft_values_rejected(self, tmp_path: Path):
        for key, value in [
            ("num_epochs", 0),
            ("learning_rate", 0.0),
            ("batch_size", 0),
            ("weight_decay", -0.1),
        ]:
            path = self._write_sft_yaml(
                tmp_path,
                extra={"sft": {"num_epochs": 2, key: value}},
                name=f"bad_sft_{key}.yaml",
            )
            with pytest.raises(ValueError):
                SynthesizerTrainingConfig.from_yaml(path)

    def test_trace_and_smoke_modes_apply(self, tmp_path: Path):
        path = self._write_sft_yaml(
            tmp_path,
            extra={
                "modes": {
                    "trace": {
                        "experiment_version": "trace",
                        "sft": {"num_epochs": 1, "batch_size": 1},
                    },
                    "smoke": {"experiment_version": "smoke"},
                }
            },
        )
        trace = SynthesizerTrainingConfig.from_yaml(path, mode="trace")
        assert trace.experiment_version == "trace"
        assert trace.sft.num_epochs == 1
        assert trace.sft.batch_size == 1
        smoke = SynthesizerTrainingConfig.from_yaml(path, mode="smoke")
        assert smoke.experiment_version == "smoke"
        assert smoke.sft.num_epochs == 2
