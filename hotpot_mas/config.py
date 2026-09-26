"""Experiment configuration.

Loads and validates the YAML config (``configs/base.yaml``), applies run-mode
overrides (smoke / small / full) and CLI overrides, and provides the resolved
configuration that is persisted with every run record (spec sec. 10, 15.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Repository root: parent directory of this package.
REPO_ROOT = Path(__file__).resolve().parent.parent

_KNOWN_KEYS = {
    "experiment_id", "experiment_version", "dataset", "dataset_config",
    "dataset_split", "num_questions", "runs_per_question", "run_seeds",
    "sample_selection_seed", "model_name", "model_revision",
    "tokenizer_revision", "dtype", "device", "attn_implementation",
    "max_input_length", "generation", "max_decision_steps", "prompt_dir",
    "manifest_path", "output_dir", "modes",
}


def _resolve_path(value: Any) -> Path:
    """Resolve a config path against the repository root unless absolute."""
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


@dataclass
class GenerationParams:
    """Decoding hyperparameters shared by all three agents (spec sec. 10)."""

    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 2048

    def to_dict(self) -> Dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
        }

    def to_generate_kwargs(self) -> Dict[str, Any]:
        return self.to_dict()


@dataclass
class ModeParams:
    """Per-mode overrides (smoke / small / full)."""

    num_questions: int
    runs_per_question: int
    run_seeds: List[int]


@dataclass
class ExperimentConfig:
    experiment_id: str = "hotpotqa_base_mas"
    experiment_version: str = "v1"
    dataset: str = "hotpot_qa"
    dataset_config: str = "distractor"
    dataset_split: str = "validation"
    num_questions: int = 100
    runs_per_question: int = 10
    run_seeds: List[int] = field(default_factory=lambda: list(range(10)))
    sample_selection_seed: int = 0
    model_name: str = "google/gemma-3-1b-it"
    model_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    dtype: str = "float16"
    device: str = "cuda"
    attn_implementation: Optional[str] = None
    max_input_length: int = 30000
    generation: GenerationParams = field(default_factory=GenerationParams)
    max_decision_steps: int = 20
    prompt_dir: Path = field(default_factory=lambda: REPO_ROOT / "prompts")
    manifest_path: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "question_manifest.json")
    output_dir: Path = field(default_factory=lambda: REPO_ROOT / "outputs")
    modes: Dict[str, ModeParams] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Any) -> "ExperimentConfig":
        """Load and validate the YAML config file."""
        raw_path = Path(str(path))
        if not raw_path.is_absolute():
            raw_path = REPO_ROOT / raw_path
        with open(raw_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        unknown = set(raw) - _KNOWN_KEYS
        if unknown:
            raise ValueError(f"unknown config keys in {raw_path}: {sorted(unknown)}")

        generation = GenerationParams(**dict(raw.get("generation", {})))
        modes = {
            name: ModeParams(
                num_questions=int(m["num_questions"]),
                runs_per_question=int(m["runs_per_question"]),
                run_seeds=[int(s) for s in m["run_seeds"]],
            )
            for name, m in (raw.get("modes") or {}).items()
        }
        cfg = cls(
            experiment_id=raw["experiment_id"],
            experiment_version=raw["experiment_version"],
            dataset=raw.get("dataset", "hotpot_qa"),
            dataset_config=raw.get("dataset_config", "distractor"),
            dataset_split=raw.get("dataset_split", "validation"),
            num_questions=int(raw.get("num_questions", 100)),
            runs_per_question=int(raw.get("runs_per_question", 10)),
            run_seeds=[int(s) for s in raw.get("run_seeds", list(range(10)))],
            sample_selection_seed=int(raw.get("sample_selection_seed", 0)),
            model_name=raw.get("model_name", "google/gemma-3-1b-it"),
            model_revision=raw.get("model_revision"),
            tokenizer_revision=raw.get("tokenizer_revision"),
            dtype=raw.get("dtype", "float16"),
            device=raw.get("device", "cuda"),
            attn_implementation=raw.get("attn_implementation"),
            max_input_length=int(raw.get("max_input_length", 30000)),
            generation=generation,
            max_decision_steps=int(raw.get("max_decision_steps", 20)),
            prompt_dir=_resolve_path(raw.get("prompt_dir", "prompts")),
            manifest_path=_resolve_path(raw.get("manifest_path", "outputs/question_manifest.json")),
            output_dir=_resolve_path(raw.get("output_dir", "outputs")),
            modes=modes,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Validate invariants that affect the number and identity of runs."""
        if self.num_questions <= 0:
            raise ValueError("num_questions must be positive")
        if self.runs_per_question <= 0:
            raise ValueError("runs_per_question must be positive")
        if not self.run_seeds:
            raise ValueError("run_seeds must not be empty")
        if len(self.run_seeds) != self.runs_per_question:
            raise ValueError(
                "runs_per_question must equal len(run_seeds): "
                f"{self.runs_per_question} != {len(self.run_seeds)}"
            )
        if len(set(self.run_seeds)) != len(self.run_seeds):
            raise ValueError("run_seeds must be unique")
        if any(seed < 0 or seed >= 2**32 for seed in self.run_seeds):
            raise ValueError("run_seeds must be in NumPy's [0, 2**32) range")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(
                "dtype must be one of: float16, bfloat16, float32"
            )
        if self.max_decision_steps <= 0:
            raise ValueError("max_decision_steps must be positive")
        if self.generation.max_new_tokens <= 0:
            raise ValueError("generation.max_new_tokens must be positive")

    def apply_mode(
        self,
        mode: Optional[str] = None,
        num_questions: Optional[int] = None,
        runs_per_question: Optional[int] = None,
        run_seeds: Optional[str] = None,
    ) -> "ExperimentConfig":
        """Return a copy with mode + CLI overrides applied."""
        cfg = self
        if mode:
            if mode not in self.modes:
                raise ValueError(f"unknown mode {mode!r}; available: {sorted(self.modes)}")
            m = self.modes[mode]
            cfg = replace(
                cfg,
                num_questions=m.num_questions,
                runs_per_question=m.runs_per_question,
                run_seeds=list(m.run_seeds),
            )
        if num_questions is not None:
            cfg = replace(cfg, num_questions=int(num_questions))
        parsed_seeds = None
        if run_seeds is not None:
            parsed_seeds = [int(s) for s in run_seeds.split(",") if s.strip()]
        if runs_per_question is not None and parsed_seeds is None:
            requested = int(runs_per_question)
            if requested > len(cfg.run_seeds):
                raise ValueError(
                    f"--runs {requested} needs at least {requested} configured seeds; "
                    f"only {len(cfg.run_seeds)} are available"
                )
            cfg = replace(
                cfg,
                runs_per_question=requested,
                run_seeds=list(cfg.run_seeds[:requested]),
            )
        elif parsed_seeds is not None and runs_per_question is None:
            cfg = replace(
                cfg,
                runs_per_question=len(parsed_seeds),
                run_seeds=parsed_seeds,
            )
        elif parsed_seeds is not None and runs_per_question is not None:
            cfg = replace(
                cfg,
                runs_per_question=int(runs_per_question),
                run_seeds=parsed_seeds,
            )
        cfg.validate()
        return cfg

    def output_subdir(self) -> Path:
        """Directory that holds runs.jsonl and the summaries."""
        return self.output_dir / self.experiment_id / self.experiment_version

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable resolved config, persisted with every run record."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_version": self.experiment_version,
            "dataset": self.dataset,
            "dataset_config": self.dataset_config,
            "dataset_split": self.dataset_split,
            "num_questions": self.num_questions,
            "runs_per_question": self.runs_per_question,
            "run_seeds": list(self.run_seeds),
            "sample_selection_seed": self.sample_selection_seed,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "dtype": self.dtype,
            "device": self.device,
            "attn_implementation": self.attn_implementation,
            "max_input_length": self.max_input_length,
            "generation": self.generation.to_dict(),
            "max_decision_steps": self.max_decision_steps,
            "prompt_dir": str(self.prompt_dir),
            "manifest_path": str(self.manifest_path),
            "output_dir": str(self.output_dir),
        }
