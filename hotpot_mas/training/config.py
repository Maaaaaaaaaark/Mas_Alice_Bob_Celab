"""Training experiment configuration (Stage 1 workers + Stage 2 synthesizer).

Deliberately separate from the inference ``ExperimentConfig``
(``hotpot_mas/config.py``): training runs have their own hyperparameters,
paths, checkpoints and run modes, and the inference baseline must stay
untouched. All values below are configurable through YAML; nothing about
the algorithm (G, N, seeds, learning rate, paths) is hardcoded.

Run modes are declared as nested YAML dictionaries under ``modes:``; a mode
is applied by deep-merging its mapping over the base configuration before
validation, so any field can be overridden per mode.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Repository root: parent directory of the package's parent (hotpot_mas).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(value: Any) -> Path:
    """Resolve a config path against the repository root unless absolute."""
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Lists and scalars in ``override`` replace the base value wholesale;
    nested dictionaries are merged key by key.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _check_seed_range(value: int, name: str) -> None:
    if value < 0 or value >= 2**32:
        raise ValueError(f"{name} must be in [0, 2**32)")


@dataclass
class SplitConfig:
    """One split's source and size."""

    split: str = "train"  # official HotpotQA split name
    num_questions: int = 100
    selection_seed: int = 0

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SplitConfig":
        return cls(
            split=str(raw["split"]),
            num_questions=int(raw["num_questions"]),
            selection_seed=int(raw["selection_seed"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "num_questions": self.num_questions,
            "selection_seed": self.selection_seed,
        }


@dataclass
class DataConfig:
    """Dataset identity plus the three training splits.

    Train, validation and test questions come from fixed manifests written
    by ``prepare-data``; question ids are asserted pairwise disjoint. The
    validation/test boundary is drawn by shuffling the selected validation
    candidates with ``val_test_split_seed``. Note: this is an
    *oracle-balanced* partition built from gold supporting-document
    metadata; it is not an agent assignment shipped with HotpotQA.
    """

    dataset: str = "hotpot_qa"
    dataset_config: str = "distractor"
    train: SplitConfig = field(default_factory=SplitConfig)
    val: SplitConfig = field(default_factory=lambda: SplitConfig(split="validation"))
    test: SplitConfig = field(default_factory=lambda: SplitConfig(split="validation"))
    partition_seed: int = 0
    val_test_split_seed: int = 0

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DataConfig":
        return cls(
            dataset=str(raw.get("dataset", "hotpot_qa")),
            dataset_config=str(raw.get("dataset_config", "distractor")),
            train=SplitConfig.from_dict(raw["train"]),
            val=SplitConfig.from_dict(raw["val"]),
            test=SplitConfig.from_dict(raw["test"]),
            partition_seed=int(raw.get("partition_seed", 0)),
            val_test_split_seed=int(raw.get("val_test_split_seed", 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "dataset_config": self.dataset_config,
            "train": self.train.to_dict(),
            "val": self.val.to_dict(),
            "test": self.test.to_dict(),
            "partition_seed": self.partition_seed,
            "val_test_split_seed": self.val_test_split_seed,
        }


@dataclass
class PeftConfig:
    """LoRA adapter hyperparameters (workers in Stage 1, C in Stage 2)."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: Optional[List[str]] = None  # None -> peft "all-linear"

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "PeftConfig":
        raw = raw or {}
        target = raw.get("target_modules")
        return cls(
            r=int(raw.get("r", 16)),
            alpha=int(raw.get("alpha", 32)),
            dropout=float(raw.get("dropout", 0.0)),
            target_modules=[str(x) for x in target] if target else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": (
                list(self.target_modules) if self.target_modules else None
            ),
        }


@dataclass
class ModelConfig:
    """Base model identity and device for the training runs."""

    name: str = "google/gemma-3-1b-it"
    revision: Optional[str] = None
    dtype: str = "float16"  # float16 | bfloat16 | float32
    device: str = "cuda"
    attn_implementation: Optional[str] = None
    max_input_length: int = 30000
    peft: PeftConfig = field(default_factory=PeftConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ModelConfig":
        return cls(
            name=str(raw["name"]),
            revision=raw.get("revision"),
            dtype=str(raw.get("dtype", "float16")),
            device=str(raw.get("device", "cuda")),
            attn_implementation=raw.get("attn_implementation"),
            max_input_length=int(raw.get("max_input_length", 30000)),
            peft=PeftConfig.from_dict(raw.get("peft")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "dtype": self.dtype,
            "device": self.device,
            "attn_implementation": self.attn_implementation,
            "max_input_length": self.max_input_length,
            "peft": self.peft.to_dict(),
        }


@dataclass
class DecodeConfig:
    """Decoding parameters for one sampling mode."""

    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 1.0
    max_new_tokens: int = 256

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DecodeConfig":
        return cls(
            do_sample=bool(raw.get("do_sample", True)),
            temperature=float(raw.get("temperature", 0.6)),
            top_p=float(raw.get("top_p", 1.0)),
            max_new_tokens=int(raw.get("max_new_tokens", 256)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass
class WorkerTrainingConfig:
    """Stage 1 hyperparameters (Algorithm 1 of cross_paired_grpo.tex)."""

    G: int = 4  # reports sampled per worker per question
    delta: float = 0.05  # signal threshold: std(Q_side) > delta
    eps_n: float = 1e-6  # advantage normalization epsilon
    clip_epsilon: float = 0.2  # PPO clip range
    num_policy_epochs: int = 1  # 1 = exactly the TeX single-update algorithm
    minibatch_size: int = 8  # questions per gradient-accumulation chunk
    max_sampling_attempts: int = 100  # guard against endless no-signal loops
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    questions_per_update: int = 8  # N: signal questions per update
    steps: int = 100  # T: number of updates
    eval_interval: int = 10  # K: validation evaluation frequency
    step0_eval: bool = False  # evaluate/save the untrained worker baseline
    final_test_eval: bool = True  # run Evaluate(theta*, C, D_test) at the end
    fail_on_insufficient_signal: bool = False
    rollout: DecodeConfig = field(default_factory=DecodeConfig)
    eval_decode: DecodeConfig = field(
        default_factory=lambda: DecodeConfig(
            do_sample=False, temperature=0.0, top_p=1.0, max_new_tokens=256
        )
    )
    synthesizer_decode: DecodeConfig = field(
        default_factory=lambda: DecodeConfig(
            do_sample=False, temperature=0.0, top_p=1.0, max_new_tokens=64
        )
    )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "WorkerTrainingConfig":
        return cls(
            G=int(raw.get("G", 4)),
            delta=float(raw.get("delta", 0.05)),
            eps_n=float(raw.get("eps_n", 1e-6)),
            clip_epsilon=float(raw.get("clip_epsilon", 0.2)),
            num_policy_epochs=int(raw.get("num_policy_epochs", 1)),
            minibatch_size=int(raw.get("minibatch_size", 8)),
            max_sampling_attempts=int(raw.get("max_sampling_attempts", 100)),
            learning_rate=float(raw.get("learning_rate", 1e-4)),
            weight_decay=float(raw.get("weight_decay", 0.0)),
            questions_per_update=int(raw.get("questions_per_update", 8)),
            steps=int(raw.get("steps", 100)),
            eval_interval=int(raw.get("eval_interval", 10)),
            step0_eval=bool(raw.get("step0_eval", False)),
            final_test_eval=bool(raw.get("final_test_eval", True)),
            fail_on_insufficient_signal=bool(
                raw.get("fail_on_insufficient_signal", False)
            ),
            rollout=DecodeConfig.from_dict(raw.get("rollout", {})),
            eval_decode=DecodeConfig.from_dict(raw.get("eval_decode", {})),
            synthesizer_decode=DecodeConfig.from_dict(
                raw.get("synthesizer_decode", {})
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "G": self.G,
            "delta": self.delta,
            "eps_n": self.eps_n,
            "clip_epsilon": self.clip_epsilon,
            "num_policy_epochs": self.num_policy_epochs,
            "minibatch_size": self.minibatch_size,
            "max_sampling_attempts": self.max_sampling_attempts,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "questions_per_update": self.questions_per_update,
            "steps": self.steps,
            "eval_interval": self.eval_interval,
            "step0_eval": self.step0_eval,
            "final_test_eval": self.final_test_eval,
            "fail_on_insufficient_signal": self.fail_on_insufficient_signal,
            "rollout": self.rollout.to_dict(),
            "eval_decode": self.eval_decode.to_dict(),
            "synthesizer_decode": self.synthesizer_decode.to_dict(),
        }


@dataclass
class TrainingConfig:
    """Stage 1 worker training configuration."""

    experiment_id: str = "cross_paired_grpo_workers"
    experiment_version: str = "v1"
    seed: int = 0
    output_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "outputs" / "training"
    )
    manifest_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "outputs" / "training_manifests"
    )
    prompt_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "prompts_training"
    )
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    workers: WorkerTrainingConfig = field(default_factory=WorkerTrainingConfig)
    modes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Any, mode: Optional[str] = None) -> "TrainingConfig":
        raw_path = Path(str(path))
        if not raw_path.is_absolute():
            raw_path = REPO_ROOT / raw_path
        with open(raw_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if mode is not None:
            raw = cls.apply_mode(raw, mode)
        cfg = cls._from_raw(raw)
        cfg.validate()
        return cfg

    @classmethod
    def _from_raw(cls, raw: Dict[str, Any]) -> "TrainingConfig":
        return cls(
            experiment_id=str(raw["experiment_id"]),
            experiment_version=str(raw["experiment_version"]),
            seed=int(raw.get("seed", 0)),
            output_dir=_resolve_path(raw.get("output_dir", "outputs/training")),
            manifest_dir=_resolve_path(
                raw.get("manifest_dir", "outputs/training_manifests")
            ),
            prompt_dir=_resolve_path(raw.get("prompt_dir", "prompts_training")),
            data=DataConfig.from_dict(raw["data"]),
            model=ModelConfig.from_dict(raw["model"]),
            workers=WorkerTrainingConfig.from_dict(raw.get("workers", {})),
            modes={
                str(name): copy.deepcopy(mode_raw)
                for name, mode_raw in (raw.get("modes") or {}).items()
            },
        )

    @staticmethod
    def apply_mode(raw: Dict[str, Any], mode: str) -> Dict[str, Any]:
        modes = raw.get("modes") or {}
        if mode not in modes:
            raise ValueError(
                f"unknown mode {mode!r}; available: {sorted(modes)}"
            )
        merged = deep_merge(raw, modes[mode])
        merged["_mode"] = mode
        return merged

    def validate(self) -> None:
        w = self.workers
        if w.G < 2:
            raise ValueError("workers.G must be >= 2 (one report per side "
                             "gives no within-side variance)")
        if w.questions_per_update < 1:
            raise ValueError("workers.questions_per_update (N) must be >= 1")
        if w.steps < 1:
            raise ValueError("workers.steps must be >= 1")
        if w.num_policy_epochs < 1:
            raise ValueError("workers.num_policy_epochs must be >= 1")
        if w.minibatch_size < 1:
            raise ValueError("workers.minibatch_size must be >= 1")
        if w.max_sampling_attempts < 1:
            raise ValueError("workers.max_sampling_attempts must be >= 1")
        if w.delta < 0:
            raise ValueError("workers.delta must be >= 0")
        if w.eps_n < 0:
            raise ValueError("workers.eps_n must be >= 0")
        if w.clip_epsilon <= 0:
            raise ValueError("workers.clip_epsilon must be > 0")
        if w.learning_rate <= 0:
            raise ValueError("workers.learning_rate must be > 0")
        if w.weight_decay < 0:
            raise ValueError("workers.weight_decay must be >= 0")
        if w.eval_interval < 1:
            raise ValueError("workers.eval_interval must be >= 1")
        for name, decode in (
            ("rollout", w.rollout),
            ("eval_decode", w.eval_decode),
            ("synthesizer_decode", w.synthesizer_decode),
        ):
            if decode.max_new_tokens < 1:
                raise ValueError(f"workers.{name}.max_new_tokens must be >= 1")
            if not 0.0 < decode.top_p <= 1.0:
                raise ValueError(f"workers.{name}.top_p must be in (0, 1]")
            if decode.do_sample and decode.temperature <= 0.0:
                raise ValueError(
                    f"workers.{name}.temperature must be > 0 when sampling"
                )
        if w.rollout.top_p != 1.0:
            raise ValueError(
                "workers.rollout.top_p must be 1.0 for GRPO so every "
                "sampled report token remains in the updated policy's "
                "probability support"
            )
        for side in (self.data.train, self.data.val, self.data.test):
            if side.num_questions < 1:
                raise ValueError(
                    f"data.{side.split}.num_questions must be >= 1"
                )
            _check_seed_range(side.selection_seed, "selection_seed")
        _check_seed_range(self.data.partition_seed, "data.partition_seed")
        _check_seed_range(
            self.data.val_test_split_seed, "data.val_test_split_seed"
        )
        _check_seed_range(self.seed, "seed")
        if self.model.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(
                "model.dtype must be one of: float16, bfloat16, float32"
            )
        if not self.model.device:
            raise ValueError("model.device must be a non-empty string")
        if self.model.max_input_length < 1:
            raise ValueError("model.max_input_length must be >= 1")
        if self.model.peft.r < 1 or self.model.peft.alpha < 1:
            raise ValueError("model.peft.r and model.peft.alpha must be >= 1")
        if not self.prompt_dir.is_dir():
            raise ValueError(f"prompt_dir does not exist: {self.prompt_dir}")

    def stage_dir(self) -> Path:
        """Checkpoint/metrics directory for this Stage 1 run."""
        return self.output_dir / self.experiment_id / self.experiment_version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_version": self.experiment_version,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "manifest_dir": str(self.manifest_dir),
            "prompt_dir": str(self.prompt_dir),
            "data": self.data.to_dict(),
            "model": self.model.to_dict(),
            "workers": self.workers.to_dict(),
        }


# --------------------------------------------------------------------------
# Stage 2: synthesizer SFT
# --------------------------------------------------------------------------

@dataclass
class SFTConfig:
    """Stage 2 SFT hyperparameters (Algorithm 2 of cross_paired_grpo.tex).

    ``worker_eval_decode`` is the deterministic worker decode used to
    produce the cached A/B reports (it must match Stage 1's
    ``workers.eval_decode`` so the reports come from the same evaluation
    setting); ``synthesizer_decode`` is C's own greedy evaluation decode.
    """

    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 4  # D_C examples per gradient-accumulation chunk
    worker_eval_decode: DecodeConfig = field(
        default_factory=lambda: DecodeConfig(
            do_sample=False, temperature=0.0, top_p=1.0, max_new_tokens=256
        )
    )
    synthesizer_decode: DecodeConfig = field(
        default_factory=lambda: DecodeConfig(
            do_sample=False, temperature=0.0, top_p=1.0, max_new_tokens=64
        )
    )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SFTConfig":
        # DecodeConfig's generic defaults are for Stage 1 rollout sampling.
        # Stage 2 must stay deterministic when either nested decode block is
        # omitted, so inherit SFTConfig's own defaults instead.
        defaults = cls()
        return cls(
            num_epochs=int(raw.get("num_epochs", 3)),
            learning_rate=float(raw.get("learning_rate", 1e-4)),
            weight_decay=float(raw.get("weight_decay", 0.0)),
            batch_size=int(raw.get("batch_size", 4)),
            worker_eval_decode=DecodeConfig.from_dict(
                raw.get(
                    "worker_eval_decode",
                    defaults.worker_eval_decode.to_dict(),
                )
            ),
            synthesizer_decode=DecodeConfig.from_dict(
                raw.get(
                    "synthesizer_decode",
                    defaults.synthesizer_decode.to_dict(),
                )
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "worker_eval_decode": self.worker_eval_decode.to_dict(),
            "synthesizer_decode": self.synthesizer_decode.to_dict(),
        }


@dataclass
class SynthesizerTrainingConfig:
    """Stage 2 configuration: SFT of C against frozen trained workers.

    ``worker_checkpoint`` points at the Stage 1 run directory (the best
    step is then resolved from the recorded validation F1, never the last
    step) or directly at a best adapter directory
    (``.../step_XXXXX/adapter``); the synthesizer gets its own LoRA ``phi``
    on the same base model.
    """

    experiment_id: str = "synthesizer_sft"
    experiment_version: str = "v1"
    seed: int = 0
    output_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "outputs" / "training"
    )
    manifest_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "outputs" / "training_manifests"
    )
    prompt_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "prompts_training"
    )
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    worker_checkpoint: Optional[str] = None  # Stage 1 run dir or adapter dir
    modes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls, path: Any, mode: Optional[str] = None
    ) -> "SynthesizerTrainingConfig":
        raw_path = Path(str(path))
        if not raw_path.is_absolute():
            raw_path = REPO_ROOT / raw_path
        with open(raw_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if mode is not None:
            raw = TrainingConfig.apply_mode(raw, mode)
        cfg = cls(
            experiment_id=str(raw["experiment_id"]),
            experiment_version=str(raw["experiment_version"]),
            seed=int(raw.get("seed", 0)),
            output_dir=_resolve_path(raw.get("output_dir", "outputs/training")),
            manifest_dir=_resolve_path(
                raw.get("manifest_dir", "outputs/training_manifests")
            ),
            prompt_dir=_resolve_path(raw.get("prompt_dir", "prompts_training")),
            data=DataConfig.from_dict(raw["data"]),
            model=ModelConfig.from_dict(raw["model"]),
            sft=SFTConfig.from_dict(raw.get("sft", {})),
            worker_checkpoint=(
                str(raw["worker_checkpoint"])
                if raw.get("worker_checkpoint")
                else None
            ),
            modes={
                str(name): copy.deepcopy(mode_raw)
                for name, mode_raw in (raw.get("modes") or {}).items()
            },
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.sft.num_epochs < 1:
            raise ValueError("sft.num_epochs must be >= 1")
        if self.sft.learning_rate <= 0:
            raise ValueError("sft.learning_rate must be > 0")
        if self.sft.weight_decay < 0:
            raise ValueError("sft.weight_decay must be >= 0")
        if self.sft.batch_size < 1:
            raise ValueError("sft.batch_size must be >= 1")
        for name, decode in (
            ("worker_eval_decode", self.sft.worker_eval_decode),
            ("synthesizer_decode", self.sft.synthesizer_decode),
        ):
            if decode.do_sample:
                raise ValueError(
                    f"sft.{name}.do_sample must be False: evaluation "
                    "decoding (workers and C) is deterministic per the TeX"
                )
            if decode.max_new_tokens < 1:
                raise ValueError(
                    f"sft.{name}.max_new_tokens must be >= 1"
                )
        if self.model.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(
                "model.dtype must be one of: float16, bfloat16, float32"
            )
        if not self.model.device:
            raise ValueError("model.device must be a non-empty string")
        if self.model.max_input_length < 1:
            raise ValueError("model.max_input_length must be >= 1")
        if self.model.peft.r < 1 or self.model.peft.alpha < 1:
            raise ValueError("model.peft.r and model.peft.alpha must be >= 1")
        _check_seed_range(self.seed, "seed")
        for side in (self.data.train, self.data.val, self.data.test):
            if side.num_questions < 1:
                raise ValueError(
                    f"data.{side.split}.num_questions must be >= 1"
                )
            _check_seed_range(side.selection_seed, "selection_seed")
        _check_seed_range(self.data.partition_seed, "data.partition_seed")
        _check_seed_range(
            self.data.val_test_split_seed, "data.val_test_split_seed"
        )
        if not self.prompt_dir.is_dir():
            raise ValueError(f"prompt_dir does not exist: {self.prompt_dir}")
        if not self.worker_checkpoint:
            raise ValueError(
                "worker_checkpoint is required: point it at the Stage 1 "
                "best adapter directory (selected on validation F1)"
            )
        if not Path(self.worker_checkpoint).is_absolute():
            self.worker_checkpoint = str(_resolve_path(self.worker_checkpoint))
        if not Path(self.worker_checkpoint).is_dir():
            raise ValueError(
                f"worker_checkpoint directory does not exist: "
                f"{self.worker_checkpoint}"
            )

    def stage_dir(self) -> Path:
        return self.output_dir / self.experiment_id / self.experiment_version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_version": self.experiment_version,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "manifest_dir": str(self.manifest_dir),
            "prompt_dir": str(self.prompt_dir),
            "data": self.data.to_dict(),
            "model": self.model.to_dict(),
            "sft": self.sft.to_dict(),
            "worker_checkpoint": self.worker_checkpoint,
        }
