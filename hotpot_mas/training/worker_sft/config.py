"""Stage 1a worker SFT configuration (``docs/WORKER_SFT.md``).

Separate from the Stage 1 GRPO and Stage 2 synthesizer configs: this
experiment trains ONE shared worker model on (local question, visible
documents) pairs with SFT, and no coordinator, RL, routing, or joint
training exists here. Run modes (trace / smoke / normal) are declared
under ``modes:`` in YAML and deep-merged over the base configuration, the
same convention as the other training configs. Every effective default is
recorded in the run's config snapshot, including weight decay and the
scheduler setting.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# The Stage-1 config module already provides the repository-root path
# helpers and the deep-merge convention; reuse them so all training
# experiments resolve paths and modes identically.
from ..config import (
    REPO_ROOT,
    _check_seed_range,
    _resolve_path,
    deep_merge,
)


@dataclass
class SourceConfig:
    """One training-data source (SQuAD 2.0 / MuSiQue / HotpotQA)."""

    name: str
    enabled: bool = True
    weight: float = 1.0  # relative mixture weight, not a quota
    max_candidates: int = 0  # cap on auto candidates; 0 = unlimited
    official_split: str = "train"

    @classmethod
    def from_dict(cls, name: str, raw: Dict[str, Any]) -> "SourceConfig":
        return cls(
            name=name,
            enabled=bool(raw.get("enabled", True)),
            weight=float(raw.get("weight", 1.0)),
            max_candidates=int(raw.get("max_candidates", 0)),
            official_split=str(raw.get("official_split", "train")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "weight": self.weight,
            "max_candidates": self.max_candidates,
            "official_split": self.official_split,
        }


@dataclass
class WorkerSFTDataConfig:
    """Data files, source switches, and the mixture targets.

    The reviewed file (``data/worker_sft_reviewed.json``) is committed in
    the repository; the optional teacher file is loaded when present and
    its absence is reported explicitly, never silently skipped. The
    MuSiQue exclusion file is the official leak-prevention list.
    """

    reviewed_file: Path = field(
        default_factory=lambda: _resolve_path("data/worker_sft_reviewed.json")
    )
    teacher_file: Optional[Path] = field(
        default_factory=lambda: _resolve_path(
            "data/local_questions_hotpot_teacher.jsonl"
        )
    )
    exclusion_file: Path = field(
        default_factory=lambda: _resolve_path(
            "data/dev_test_singlehop_questions_v1.0.json"
        )
    )
    exclusion_sha256: str = (
        "013be00ab914799891e5ae40de6cdc1baf03aaffd1f3ce4bf0de796020069613"
    )
    eval_manifest_paths: List[Path] = field(
        default_factory=lambda: [
            _resolve_path("outputs/question_manifest_distractor.json"),
            _resolve_path("outputs/hotpotqa_base_mas/manifest.json"),
        ]
    )
    sources: Dict[str, SourceConfig] = field(
        default_factory=lambda: {
            "squad2": SourceConfig(name="squad2", max_candidates=5000),
            "musique": SourceConfig(name="musique", max_candidates=1500),
            # HotpotQA training examples come only from the reviewed (and
            # optional teacher) files; there is no live teacher call.
            "hotpot": SourceConfig(name="hotpot", max_candidates=0),
        }
    )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "WorkerSFTDataConfig":
        sources = {}
        for name, source_raw in (raw.get("sources") or {}).items():
            sources[str(name)] = SourceConfig.from_dict(
                str(name), source_raw or {}
            )
        defaults = cls()
        return cls(
            reviewed_file=_resolve_path(
                raw.get("reviewed_file", defaults.reviewed_file)
            ),
            teacher_file=(
                _resolve_path(raw["teacher_file"])
                if raw.get("teacher_file")
                else None
            ),
            exclusion_file=_resolve_path(
                raw.get("exclusion_file", defaults.exclusion_file)
            ),
            exclusion_sha256=str(
                raw.get("exclusion_sha256", defaults.exclusion_sha256)
            ),
            eval_manifest_paths=[
                _resolve_path(p)
                for p in raw.get(
                    "eval_manifest_paths",
                    [str(p) for p in defaults.eval_manifest_paths],
                )
            ],
            sources=sources,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reviewed_file": str(self.reviewed_file),
            "teacher_file": (
                str(self.teacher_file) if self.teacher_file else None
            ),
            "exclusion_file": str(self.exclusion_file),
            "exclusion_sha256": self.exclusion_sha256,
            "eval_manifest_paths": [str(p) for p in self.eval_manifest_paths],
            "sources": {
                name: source.to_dict()
                for name, source in self.sources.items()
            },
        }


@dataclass
class WorkerSFTMixConfig:
    """Mixture targets and dedup thresholds."""

    target_answerable_ratio: float = 0.7
    near_dup_question_jaccard: float = 0.9
    near_dup_paragraph_jaccard: float = 0.95
    val_test_freeze_seed: int = 1234  # val/test definitions are frozen first
    val_parents: int = 4  # parents frozen into val (model selection)
    test_parents: int = 4  # parents frozen into test (final evaluation)
    total_examples: int = 0  # mode-specific totals live in modes:
    trace_num_examples: int = 10
    trace_positives: int = 7
    trace_negatives: int = 3
    selection_seed: int = 0

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "WorkerSFTMixConfig":
        defaults = cls()
        return cls(
            target_answerable_ratio=float(
                raw.get("target_answerable_ratio", 0.7)
            ),
            near_dup_question_jaccard=float(
                raw.get("near_dup_question_jaccard", 0.9)
            ),
            near_dup_paragraph_jaccard=float(
                raw.get("near_dup_paragraph_jaccard", 0.95)
            ),
            val_test_freeze_seed=int(raw.get("val_test_freeze_seed", 1234)),
            val_parents=int(raw.get("val_parents", 4)),
            test_parents=int(raw.get("test_parents", 4)),
            total_examples=int(raw.get("total_examples", 0)),
            trace_num_examples=int(raw.get("trace_num_examples", 10)),
            trace_positives=int(raw.get("trace_positives", 7)),
            trace_negatives=int(raw.get("trace_negatives", 3)),
            selection_seed=int(raw.get("selection_seed", 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_answerable_ratio": self.target_answerable_ratio,
            "near_dup_question_jaccard": self.near_dup_question_jaccard,
            "near_dup_paragraph_jaccard": self.near_dup_paragraph_jaccard,
            "val_test_freeze_seed": self.val_test_freeze_seed,
            "val_parents": self.val_parents,
            "test_parents": self.test_parents,
            "total_examples": self.total_examples,
            "trace_num_examples": self.trace_num_examples,
            "trace_positives": self.trace_positives,
            "trace_negatives": self.trace_negatives,
            "selection_seed": self.selection_seed,
        }


@dataclass
class WorkerSFTPeftConfig:
    """LoRA adapter for the worker. A FRESH adapter is always created;
    no old GRPO/synthesizer adapter is ever loaded."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # None -> "all-linear"

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "WorkerSFTPeftConfig":
        defaults = cls()
        target = raw.get("target_modules")
        return cls(
            r=int(raw.get("r", defaults.r)),
            alpha=int(raw.get("alpha", defaults.alpha)),
            dropout=float(raw.get("dropout", defaults.dropout)),
            target_modules=(
                [str(x) for x in target] if target is not None else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
        }


@dataclass
class WorkerSFTModelConfig:
    """Base model identity and precision for the worker."""

    name: str = "google/gemma-3-1b-it"
    revision: Optional[str] = None  # pinned at data-prep time and recorded
    dtype: str = "bfloat16"  # bf16 when supported, fp16 fallback recorded
    device: str = "cuda"
    max_input_tokens: int = 8192  # never truncate silently (hard error)
    peft: WorkerSFTPeftConfig = field(default_factory=WorkerSFTPeftConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "WorkerSFTModelConfig":
        defaults = cls()
        return cls(
            name=str(raw.get("name", defaults.name)),
            revision=raw.get("revision"),
            dtype=str(raw.get("dtype", defaults.dtype)),
            device=str(raw.get("device", defaults.device)),
            max_input_tokens=int(
                raw.get("max_input_tokens", defaults.max_input_tokens)
            ),
            peft=WorkerSFTPeftConfig.from_dict(raw.get("peft", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "dtype": self.dtype,
            "device": self.device,
            "max_input_tokens": self.max_input_tokens,
            "peft": self.peft.to_dict(),
        }


@dataclass
class WorkerSFTTrainConfig:
    """SFT hyperparameters. All effective values are recorded per run.

    ``scheduler`` is "none" by default: with a single epoch and a fixed
    learning rate there is no LR schedule, and that fact is written into
    the run's config snapshot rather than assumed.
    """

    num_epochs: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    adam_betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    adam_eps: float = 1e-8
    microbatch_size: int = 1
    grad_accum_steps: int = 8  # the later normal run; trace sets 1
    max_grad_norm: float = 1.0
    scheduler: str = "none"  # none | linear
    warmup_ratio: float = 0.0
    seed: int = 0
    example_order_seed: int = 0

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "WorkerSFTTrainConfig":
        defaults = cls()
        return cls(
            num_epochs=int(raw.get("num_epochs", defaults.num_epochs)),
            learning_rate=float(
                raw.get("learning_rate", defaults.learning_rate)
            ),
            weight_decay=float(raw.get("weight_decay", defaults.weight_decay)),
            adam_betas=[
                float(x) for x in raw.get("adam_betas", defaults.adam_betas)
            ],
            adam_eps=float(raw.get("adam_eps", defaults.adam_eps)),
            microbatch_size=int(
                raw.get("microbatch_size", defaults.microbatch_size)
            ),
            grad_accum_steps=int(
                raw.get("grad_accum_steps", defaults.grad_accum_steps)
            ),
            max_grad_norm=float(
                raw.get("max_grad_norm", defaults.max_grad_norm)
            ),
            scheduler=str(raw.get("scheduler", defaults.scheduler)),
            warmup_ratio=float(raw.get("warmup_ratio", defaults.warmup_ratio)),
            seed=int(raw.get("seed", defaults.seed)),
            example_order_seed=int(
                raw.get("example_order_seed", defaults.example_order_seed)
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "adam_betas": self.adam_betas,
            "adam_eps": self.adam_eps,
            "microbatch_size": self.microbatch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "max_grad_norm": self.max_grad_norm,
            "scheduler": self.scheduler,
            "warmup_ratio": self.warmup_ratio,
            "seed": self.seed,
            "example_order_seed": self.example_order_seed,
        }


@dataclass
class WorkerSFTDecodeConfig:
    """Deterministic greedy decoding for the fixed pre/post probes."""

    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 128
    probe_seed: int = 12345

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "WorkerSFTDecodeConfig":
        defaults = cls()
        return cls(
            do_sample=bool(raw.get("do_sample", defaults.do_sample)),
            temperature=float(raw.get("temperature", defaults.temperature)),
            top_p=float(raw.get("top_p", defaults.top_p)),
            max_new_tokens=int(
                raw.get("max_new_tokens", defaults.max_new_tokens)
            ),
            probe_seed=int(raw.get("probe_seed", defaults.probe_seed)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "probe_seed": self.probe_seed,
        }


@dataclass
class WorkerSFTReviewConfig:
    """Label-quality gate: which examples may enter training."""

    require_reviewed_for_training: bool = False  # true in trace mode

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "WorkerSFTReviewConfig":
        defaults = cls()
        return cls(
            require_reviewed_for_training=bool(
                raw.get(
                    "require_reviewed_for_training",
                    defaults.require_reviewed_for_training,
                )
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "require_reviewed_for_training": self.require_reviewed_for_training,
        }


@dataclass
class WorkerSFTConfig:
    """Stage 1a worker SFT configuration root."""

    experiment_id: str = "worker_sft"
    experiment_version: str = "v0"
    seed: int = 0
    output_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "outputs" / "worker_sft"
    )
    data_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "outputs" / "worker_sft_data"
    )
    prompt_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "prompts_worker_sft"
    )
    data: WorkerSFTDataConfig = field(default_factory=WorkerSFTDataConfig)
    mix: WorkerSFTMixConfig = field(default_factory=WorkerSFTMixConfig)
    model: WorkerSFTModelConfig = field(default_factory=WorkerSFTModelConfig)
    sft: WorkerSFTTrainConfig = field(default_factory=WorkerSFTTrainConfig)
    decode: WorkerSFTDecodeConfig = field(
        default_factory=WorkerSFTDecodeConfig
    )
    review: WorkerSFTReviewConfig = field(
        default_factory=WorkerSFTReviewConfig
    )
    modes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls, path: Any, mode: Optional[str] = None
    ) -> "WorkerSFTConfig":
        raw_path = Path(str(path))
        if not raw_path.is_absolute():
            raw_path = REPO_ROOT / raw_path
        with open(raw_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if mode is not None:
            raw = cls.apply_mode(raw, mode)
        cfg = cls(
            experiment_id=str(raw["experiment_id"]),
            experiment_version=str(raw["experiment_version"]),
            seed=int(raw.get("seed", 0)),
            output_dir=_resolve_path(
                raw.get("output_dir", "outputs/worker_sft")
            ),
            data_dir=_resolve_path(
                raw.get("data_dir", "outputs/worker_sft_data")
            ),
            prompt_dir=_resolve_path(
                raw.get("prompt_dir", "prompts_worker_sft")
            ),
            data=WorkerSFTDataConfig.from_dict(raw.get("data", {})),
            mix=WorkerSFTMixConfig.from_dict(raw.get("mix", {})),
            model=WorkerSFTModelConfig.from_dict(raw.get("model", {})),
            sft=WorkerSFTTrainConfig.from_dict(raw.get("sft", {})),
            decode=WorkerSFTDecodeConfig.from_dict(raw.get("decode", {})),
            review=WorkerSFTReviewConfig.from_dict(raw.get("review", {})),
            modes={
                str(name): copy.deepcopy(mode_raw)
                for name, mode_raw in (raw.get("modes") or {}).items()
            },
        )
        cfg.validate()
        return cfg

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
        if self.sft.num_epochs < 1:
            raise ValueError("sft.num_epochs must be >= 1")
        if self.sft.learning_rate <= 0:
            raise ValueError("sft.learning_rate must be > 0")
        if self.sft.weight_decay < 0:
            raise ValueError("sft.weight_decay must be >= 0")
        if self.sft.microbatch_size < 1:
            raise ValueError("sft.microbatch_size must be >= 1")
        if self.sft.grad_accum_steps < 1:
            raise ValueError("sft.grad_accum_steps must be >= 1")
        if self.sft.max_grad_norm <= 0:
            raise ValueError("sft.max_grad_norm must be > 0")
        if self.sft.scheduler not in {"none", "linear"}:
            raise ValueError("sft.scheduler must be none or linear")
        if self.sft.warmup_ratio < 0 or self.sft.warmup_ratio >= 1:
            raise ValueError("sft.warmup_ratio must be in [0, 1)")
        if not 0.0 <= self.mix.target_answerable_ratio <= 1.0:
            raise ValueError("mix.target_answerable_ratio must be in [0, 1]")
        if self.model.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(
                "model.dtype must be one of: float16, bfloat16, float32"
            )
        if self.model.max_input_tokens < 1:
            raise ValueError("model.max_input_tokens must be >= 1")
        if self.model.peft.r < 1 or self.model.peft.alpha < 1:
            raise ValueError("model.peft.r and model.peft.alpha must be >= 1")
        if self.model.peft.dropout < 0 or self.model.peft.dropout >= 1:
            raise ValueError("model.peft.dropout must be in [0, 1)")
        if not self.decode.do_sample:
            # Greedy probes are fixed; temperature/top_p are recorded but
            # not used, so any positive value is acceptable.
            pass
        if self.decode.max_new_tokens < 1:
            raise ValueError("decode.max_new_tokens must be >= 1")
        for name in ("squad2", "musique", "hotpot"):
            if name not in self.data.sources:
                raise ValueError(f"data.sources must declare {name!r}")
        for source in self.data.sources.values():
            if source.weight < 0:
                raise ValueError(
                    f"data.sources.{source.name}.weight must be >= 0"
                )
        if not self.data.exclusion_file.is_file():
            raise ValueError(
                "data.exclusion_file does not exist: "
                f"{self.data.exclusion_file}"
            )
        if not self.data.reviewed_file.is_file():
            raise ValueError(
                "data.reviewed_file does not exist: "
                f"{self.data.reviewed_file}"
            )
        if not self.prompt_dir.is_dir():
            raise ValueError(f"prompt_dir does not exist: {self.prompt_dir}")
        _check_seed_range(self.seed, "seed")
        _check_seed_range(self.sft.seed, "sft.seed")
        _check_seed_range(self.sft.example_order_seed, "example_order_seed")
        _check_seed_range(self.mix.selection_seed, "mix.selection_seed")
        _check_seed_range(
            self.mix.val_test_freeze_seed, "mix.val_test_freeze_seed"
        )

    def prepared_data_dir(self) -> Path:
        """Where prepare-worker-sft-data writes candidates/splits."""
        return self.data_dir / self.experiment_version

    def stage_dir(self) -> Path:
        """Checkpoint/metrics/trace directory for this run."""
        return self.output_dir / self.experiment_id / self.experiment_version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_version": self.experiment_version,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "data_dir": str(self.data_dir),
            "prompt_dir": str(self.prompt_dir),
            "data": self.data.to_dict(),
            "mix": self.mix.to_dict(),
            "model": self.model.to_dict(),
            "sft": self.sft.to_dict(),
            "decode": self.decode.to_dict(),
            "review": self.review.to_dict(),
        }
