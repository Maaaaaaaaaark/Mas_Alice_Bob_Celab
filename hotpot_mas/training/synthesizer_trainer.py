"""Stage 2: SFT of the synthesizer C against frozen trained workers
(Algorithm 2 of ``cross_paired_grpo.tex``).

Loop outline:

1. Resolve the **best** Stage 1 worker checkpoint from the recorded
   validation F1 (``metrics.jsonl`` argmax cross-checked against the final
   ``state.json``) — never the latest step. Load it into the worker policy
   and freeze every parameter; the two workers keep sharing one LoRA and
   differ only in role prompt and private evidence.
2. Deterministically decode one A and one B report per question with the
   worker evaluation decoding (``do_sample=False``), and cache them on disk
   keyed by checkpoint, split, decode parameters, prompt version and
   template hashes. C0 and Cphi read the **same** cache, and the training
   cache is separate from the val/test evaluation cache.
3. Release the worker policy before loading C (single GPU).
4. Evaluate the freshly initialized C (zero-init LoRA == base model) at
   step 0: the C0 baseline on validation and test, with and without
   reports (Evaluate_empty control).
5. SFT-train C's LoRA ``phi`` only: AdamW minibatches over D_C built from
   the cached reports, cross-entropy on the **gold-answer tokens only**
   (question, reports and the ``Celab: <FINAL>`` wrapper are masked),
   mean over each example's answer tokens (the TeX's ``1/|y*|``).
6. After each epoch, evaluate C on the fixed validation report cache;
   keep the checkpoint with the best validation F1 and restore it at the
   end (never the last epoch).
7. Final comparison on the identical cached test reports: C0 + reports,
   Cphi + reports, C0 + empty, Cphi + empty.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from hotpot_mas.logging_io import JsonlWriter, collect_environment_info
from hotpot_mas.seeds import seed_all

from .config import SynthesizerTrainingConfig
from .data import describe_split, load_questions_for_split
from .eval import (
    EvalReportCache,
    SplitEvalResult,
    cache_identity,
    evaluate_synthesizer_on_cache,
    generate_report_cache,
)
from .policy import HFPolicy, Policy
from .prompts_builder import TrainingPrompts
from .sft_data import build_sft_example, collate_sft_examples
from .sft_loss import (
    assert_finite,
    assert_finite_float,
    batch_sft_loss,
)
from .synthesizers import (
    HFLoraSynthesizer,
    TrainableSynthesizer,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Stage 1 best-checkpoint resolution
# ----------------------------------------------------------------------

@dataclass
class WorkerCheckpoint:
    """The resolved best Stage 1 worker checkpoint."""

    requested: Path  # what the config asked for
    adapter_dir: Path  # where the worker LoRA lives
    best_step: Optional[int]  # None when only the adapter dir was given
    best_val_f1: Optional[float]  # None when only the adapter dir was given
    run_dir: Optional[Path]  # Stage 1 run dir when one was resolved
    identity: str  # stable id for the report-cache key
    adapter_config: Optional[Dict[str, Any]]  # peft adapter_config.json
    resolution: str  # "adapter_dir" | "metrics_jsonl" | "state_jsonl"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return lines


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _adapter_config(adapter_dir: Path) -> Optional[Dict[str, Any]]:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        return None
    return _read_json(config_path)


def resolve_worker_checkpoint(path: Any) -> WorkerCheckpoint:
    """Resolve the best Stage 1 worker checkpoint, never the latest step.

    ``path`` may be the Stage 1 run directory (the best step is resolved
    from the recorded validation F1) or a best adapter directory directly
    (``.../step_XXXXX/adapter``). Resolution from a run directory uses the
    ``metrics.jsonl`` ``mean_val_f1`` argmax and cross-checks it against the
    final ``state.json`` (``best_step``/``best_val_f1``) with exact
    equality; any conflict raises. If ``metrics.jsonl`` is absent, the
    final ``state.json`` alone may resolve the checkpoint. Failures raise
    instead of silently falling back to an untrained worker.
    """
    requested = Path(path)
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    if not requested.is_dir():
        raise FileNotFoundError(
            f"worker_checkpoint directory not found: {requested}"
        )

    # Fast path: the path is already an adapter directory.
    if (requested / "adapter_config.json").is_file() or requested.name == "adapter":
        adapter_dir = requested
        best_step: Optional[int] = None
        best_val_f1: Optional[float] = None
        parent = adapter_dir.parent.name
        if parent.startswith("step_"):
            try:
                best_step = int(parent.split("_")[1])
            except (IndexError, ValueError):
                best_step = None
        candidate_run_dir = adapter_dir.parent.parent
        run_dir = None
        if candidate_run_dir.is_dir() and (
            (candidate_run_dir / "metrics.jsonl").is_file()
            or list(candidate_run_dir.glob("step_*"))
        ):
            run_dir = candidate_run_dir
        identity = (
            f"step_{best_step:05d}"
            if best_step is not None
            else "adapter-" + hashlib.sha256(
                str(adapter_dir).encode("utf-8")
            ).hexdigest()[:12]
        )
        return WorkerCheckpoint(
            requested=requested,
            adapter_dir=adapter_dir,
            best_step=best_step,
            best_val_f1=best_val_f1,
            run_dir=run_dir,
            identity=identity,
            adapter_config=_adapter_config(adapter_dir),
            resolution="adapter_dir",
        )

    run_dir = requested
    metrics_path = run_dir / "metrics.jsonl"
    step_dirs = sorted(run_dir.glob("step_*"))
    if not step_dirs:
        raise RuntimeError(
            f"no step_* checkpoint directories in worker run dir "
            f"{run_dir}; cannot resolve a best worker checkpoint"
        )

    final_state_path = step_dirs[-1] / "state.json"
    final_state: Optional[Dict[str, Any]] = None
    if final_state_path.is_file():
        final_state = _read_json(final_state_path)

    resolution = ""
    best_step: Optional[int] = None
    best_val_f1: Optional[float] = None

    if metrics_path.is_file():
        resolution = "metrics_jsonl"
        best_update: Optional[int] = None
        best_f1 = float("-inf")
        for record in _read_jsonl(metrics_path):
            f1 = record.get("mean_val_f1")
            if f1 is None:
                continue
            if not isinstance(f1, (int, float)):
                raise RuntimeError(
                    f"non-numeric mean_val_f1 in {metrics_path}: {f1!r}"
                )
            f1 = float(f1)
            update = record.get("update")
            if update is None:
                continue
            if f1 > best_f1:
                best_f1 = f1
                best_update = int(update)
        if best_update is None:
            raise RuntimeError(
                f"metrics.jsonl in {run_dir} records no validation F1 "
                "(a trace-only Stage 1 run has no best checkpoint by "
                "validation F1); rerun Stage 1 without trace mode first"
            )
        best_step = best_update
        best_val_f1 = best_f1

        if final_state is not None:
            state_best_step = final_state.get("best_step")
            state_best_f1 = final_state.get("best_val_f1")
            if state_best_step is None:
                raise RuntimeError(
                    f"final state.json in {run_dir} has no best_step while "
                    "metrics.jsonl resolves one; the Stage 1 run is "
                    "inconsistent and cannot supply a best checkpoint"
                )
            if int(state_best_step) != best_step:
                raise RuntimeError(
                    "worker checkpoint conflict: metrics.jsonl best update "
                    f"is step {best_step} but the final state.json records "
                    f"best_step {state_best_step}"
                )
            if state_best_f1 is not None and float(state_best_f1) != best_f1:
                raise RuntimeError(
                    "worker checkpoint conflict: metrics.jsonl best "
                    f"mean_val_f1 is {best_f1!r} but the final state.json "
                    f"records best_val_f1 {state_best_f1!r}"
                )
    elif final_state is not None and final_state.get("best_step") is not None:
        resolution = "state_jsonl"
        best_step = int(final_state["best_step"])
        best_val_f1 = float(final_state["best_val_f1"])
    else:
        raise RuntimeError(
            f"cannot resolve a best worker checkpoint from {run_dir}: no "
            "metrics.jsonl and no state.json with a best_step; a Stage 1 "
            "run without validation evaluation has no best checkpoint"
        )

    adapter_dir = run_dir / f"step_{best_step:05d}" / "adapter"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"best worker adapter directory not found: {adapter_dir} "
            "(the checkpoint is incomplete); refusing to fall back to an "
            "untrained worker"
        )
    return WorkerCheckpoint(
        requested=requested,
        adapter_dir=adapter_dir,
        best_step=best_step,
        best_val_f1=best_val_f1,
        run_dir=run_dir,
        identity=f"step_{best_step:05d}",
        adapter_config=_adapter_config(adapter_dir),
        resolution=resolution,
    )


def check_worker_adapter_compatibility(
    checkpoint: WorkerCheckpoint, cfg: SynthesizerTrainingConfig
) -> Optional[Dict[str, Any]]:
    """Verify the worker adapter was trained on the configured base/LoRA.

    Compares ``adapter_config.json`` (base model name, r, lora_alpha)
    against the Stage 2 model configuration. Returns the compat summary,
    or ``None`` when no ``adapter_config.json`` is present (e.g. test
    doubles) and nothing can be checked.
    """
    adapter_config = checkpoint.adapter_config
    if not adapter_config:
        return None
    base = str(adapter_config.get("base_model_name_or_path", ""))
    r = adapter_config.get("r")
    alpha = adapter_config.get("lora_alpha")
    problems: List[str] = []
    if base != cfg.model.name:
        problems.append(
            f"adapter base model {base!r} != configured {cfg.model.name!r}"
        )
    if r is not None and int(r) != cfg.model.peft.r:
        problems.append(
            f"adapter r {r} != configured model.peft.r {cfg.model.peft.r}"
        )
    if alpha is not None and int(alpha) != cfg.model.peft.alpha:
        problems.append(
            f"adapter lora_alpha {alpha} != configured "
            f"model.peft.alpha {cfg.model.peft.alpha}"
        )
    if problems:
        raise RuntimeError(
            "worker adapter is incompatible with the Stage 2 model "
            "configuration: " + "; ".join(problems)
        )
    return {
        "checked": True,
        "base_model_name_or_path": base,
        "r": r,
        "lora_alpha": alpha,
    }


# ----------------------------------------------------------------------
# trainer
# ----------------------------------------------------------------------

class SynthesizerTrainer:
    """SFT of C's LoRA ``phi`` on gold-answer tokens against frozen workers.

    ``policy_factory`` and ``synthesizer_factory`` are zero-argument
    callables; the defaults build the real HF models. Tests pass the
    ``FakePolicy`` / ``FakeTrainableSynthesizer`` doubles.
    """

    def __init__(
        self,
        cfg: SynthesizerTrainingConfig,
        mode: str,
        policy_factory: Optional[Callable[[], Policy]] = None,
        synthesizer_factory: Optional[Callable[[], TrainableSynthesizer]] = None,
    ):
        self.cfg = cfg
        self.mode = mode
        self.stage_dir = cfg.stage_dir()
        self.stage_dir.mkdir(parents=True, exist_ok=True)

        # Fresh-run protection: refuse to reuse a directory with artifacts.
        artifacts = [
            self.stage_dir / "metrics.jsonl",
            self.stage_dir / "step0_baseline.json",
            self.stage_dir / "final_comparison.json",
            self.stage_dir / "report.md",
            self.stage_dir / "best_checkpoint",
        ] + sorted(self.stage_dir.glob("epoch_*"))
        existing = [str(p) for p in artifacts if p.exists()]
        if existing:
            raise RuntimeError(
                "refusing to start: the Stage 2 output directory already "
                f"contains artifacts from a previous run ({', '.join(existing)}); "
                "remove the directory or choose another experiment_version"
            )

        seed_all(cfg.seed)
        self.prompts = TrainingPrompts(cfg.prompt_dir)
        self.environment = collect_environment_info()

        logger.info("resolving Stage 1 best worker checkpoint from %s",
                    cfg.worker_checkpoint)
        self.worker_checkpoint = resolve_worker_checkpoint(
            cfg.worker_checkpoint
        )
        compat = check_worker_adapter_compatibility(self.worker_checkpoint, cfg)
        logger.info(
            "worker checkpoint: adapter=%s best_step=%s best_val_f1=%s "
            "resolution=%s",
            self.worker_checkpoint.adapter_dir,
            self.worker_checkpoint.best_step,
            self.worker_checkpoint.best_val_f1,
            self.worker_checkpoint.resolution,
        )

        self.train_questions = load_questions_for_split(cfg, "train")
        self.val_questions = load_questions_for_split(cfg, "val")
        self.test_questions = load_questions_for_split(cfg, "test")
        logger.info("Stage 2 %s: %s", mode, describe_split(cfg))

        # 1. Load the best worker checkpoint and freeze it.
        self.worker_policy = (
            policy_factory() if policy_factory else HFPolicy(cfg.model)
        )
        self.worker_policy.load_adapter(self.worker_checkpoint.adapter_dir)
        self._freeze_worker_policy()
        if any(
            param.requires_grad
            for param in self.worker_policy.trainable_parameters()
        ):
            raise RuntimeError(
                "worker policy still has trainable parameters after "
                "freezing; the workers must be frozen in Stage 2"
            )

        # 2. Deterministic A/B report caches per split (train, val, test).
        self.report_caches: Dict[str, EvalReportCache] = {}
        self.report_cache_keys: Dict[str, str] = {}
        cache_dir = self.stage_dir / "report_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for split, questions in (
            ("train", self.train_questions),
            ("val", self.val_questions),
            ("test", self.test_questions),
        ):
            key = cache_identity(
                self.worker_checkpoint.identity,
                split,
                cfg.sft.worker_eval_decode,
                self.prompts,
                cfg.prompt_dir,
            )
            cache = EvalReportCache(cache_dir / f"{key}.json")
            generate_report_cache(
                self.worker_policy,
                questions,
                self.prompts,
                cfg.sft.worker_eval_decode,
                cfg.seed,
                cache,
            )
            self.report_caches[split] = cache
            self.report_cache_keys[split] = key
            logger.info(
                "report cache %s: %s (%d questions)",
                split, cache.cache_path, len(questions),
            )

        self._write_report_cache_metadata()
        self._write_provenance(compat)

        # 3. Release the worker before loading C (single GPU).
        self._release_worker_policy()

        # 4. C with its own fresh zero-init LoRA == the C0 base model.
        self.synthesizer: TrainableSynthesizer = (
            synthesizer_factory()
            if synthesizer_factory
            else HFLoraSynthesizer(cfg.model, cfg.sft.synthesizer_decode)
        )
        self.optimizer = torch.optim.AdamW(
            self.synthesizer.trainable_parameters(),
            lr=cfg.sft.learning_rate,
            weight_decay=cfg.sft.weight_decay,
        )
        self.metrics_path = self.stage_dir / "metrics.jsonl"
        self.metrics = JsonlWriter(self.metrics_path)
        self.best_val_f1 = -1.0
        self.best_epoch: Optional[int] = None

    # -- worker lifecycle ----------------------------------------------

    def _freeze_worker_policy(self) -> None:
        for param in self.worker_policy.trainable_parameters():
            param.requires_grad_(False)

    def _release_worker_policy(self) -> None:
        del self.worker_policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- provenance ----------------------------------------------------

    def _write_report_cache_metadata(self) -> None:
        metadata: Dict[str, Any] = {
            "worker_checkpoint_identity": self.worker_checkpoint.identity,
            "worker_eval_decode": self.cfg.sft.worker_eval_decode.to_dict(),
            "splits": {},
        }
        for split, cache in self.report_caches.items():
            entries = cache._data
            a_tokens = [
                len(entry.get("a_ids", [])) for entry in entries.values()
            ]
            b_tokens = [
                len(entry.get("b_ids", [])) for entry in entries.values()
            ]
            metadata["splits"][split] = {
                "cache_identity": self.report_cache_keys[split],
                "path": str(cache.cache_path),
                "num_questions": len(entries),
                "mean_a_tokens": sum(a_tokens) / len(a_tokens)
                if a_tokens else 0.0,
                "mean_b_tokens": sum(b_tokens) / len(b_tokens)
                if b_tokens else 0.0,
            }
        (self.stage_dir / "report_cache_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_provenance(
        self, compat: Optional[Dict[str, Any]]
    ) -> None:
        (self.stage_dir / "worker_checkpoint_provenance.json").write_text(
            json.dumps(
                {
                    "requested": str(self.worker_checkpoint.requested),
                    "adapter_dir": str(self.worker_checkpoint.adapter_dir),
                    "run_dir": str(self.worker_checkpoint.run_dir)
                    if self.worker_checkpoint.run_dir else None,
                    "best_step": self.worker_checkpoint.best_step,
                    "best_val_f1": self.worker_checkpoint.best_val_f1,
                    "identity": self.worker_checkpoint.identity,
                    "resolution": self.worker_checkpoint.resolution,
                    "adapter_compatibility": compat,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (self.stage_dir / "config_snapshot.json").write_text(
            json.dumps(
                {
                    "mode": self.mode,
                    "config": self.cfg.to_dict(),
                    "environment": self.environment,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # -- SFT data ------------------------------------------------------

    def _build_sft_examples(self) -> List[Any]:
        """D_C: one example per training question from the cached reports."""
        cache = self.report_caches["train"]
        examples = []
        for question in self.train_questions:
            entry = cache.get(question.question_id)
            if entry is None:
                raise RuntimeError(
                    f"no cached reports for training question "
                    f"{question.question_id}; run worker evaluation first"
                )
            messages = self.prompts.synthesizer_messages(
                question.question, entry["a_text"], entry["b_text"]
            )
            example = build_sft_example(
                self.synthesizer.tokenizer,
                messages,
                question.answer,
                question_id=question.question_id,
            )
            if len(example.input_ids) > self.cfg.model.max_input_length:
                raise RuntimeError(
                    f"SFT example {question.question_id} has "
                    f"{len(example.input_ids)} tokens, exceeding "
                    f"max_input_length {self.cfg.model.max_input_length}"
                )
            examples.append(example)
        logger.info(
            "built %d SFT examples (train questions)", len(examples)
        )
        return examples

    # -- evaluation ----------------------------------------------------

    def _evaluate_split(
        self, split: str, empty_control: bool = False
    ) -> SplitEvalResult:
        questions = {
            "train": self.train_questions,
            "val": self.val_questions,
            "test": self.test_questions,
        }[split]
        cache = self.report_caches[split]
        return evaluate_synthesizer_on_cache(
            self.synthesizer,
            questions,
            self.prompts,
            cache,
            split,
            empty_control=empty_control,
        )

    # -- training ------------------------------------------------------

    def _grad_norm(self) -> float:
        total = 0.0
        for param in self.synthesizer.trainable_parameters():
            if param.grad is not None:
                total += float(param.grad.detach().float().norm().item()) ** 2
        return total**0.5

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """One full pass over D_C; returns (mean train loss, grad norm)."""
        examples = self._build_sft_examples()
        rng = random.Random(self.cfg.seed + epoch)
        rng.shuffle(examples)
        pad_token_id = getattr(
            self.synthesizer.tokenizer, "pad_token_id", None
        )
        if pad_token_id is None:
            pad_token_id = getattr(
                self.synthesizer.tokenizer, "eos_token_id", 0
            )
        pad_token_id = int(pad_token_id)
        device = self.cfg.model.device
        batch_size = self.cfg.sft.batch_size
        losses: List[float] = []
        grad_norms: List[float] = []
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            input_ids, attention_mask, labels = collate_sft_examples(
                chunk, pad_token_id
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = self.synthesizer.sft_logits(input_ids, attention_mask)
            loss = batch_sft_loss(logits, labels)
            assert_finite(loss, "SFT loss")
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = self._grad_norm()
            assert_finite_float(grad_norm, "grad norm")
            self.optimizer.step()
            losses.append(float(loss.detach().item()))
            grad_norms.append(grad_norm)
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        assert_finite_float(mean_loss, "train loss")
        assert_finite_float(grad_norm, "grad norm")
        return mean_loss, grad_norm

    # -- checkpoints ---------------------------------------------------

    def _save_checkpoint(
        self, dir_name: str, epoch: int, best_val_f1: float
    ) -> Path:
        ckpt_dir = self.stage_dir / dir_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.synthesizer.save_adapter(ckpt_dir / "adapter")
        (ckpt_dir / "state.json").write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "best_val_f1": best_val_f1,
                    "best_epoch": self.best_epoch,
                    "mode": self.mode,
                    "seed": self.cfg.seed,
                    "num_epochs": self.cfg.sft.num_epochs,
                    "num_train_questions": len(self.train_questions),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return ckpt_dir

    # -- metrics -------------------------------------------------------

    def _mean_c_tokens(self, result: SplitEvalResult) -> float:
        if not result.questions:
            return 0.0
        return sum(q.c_generated_tokens for q in result.questions) / len(
            result.questions
        )

    def _parsed_counts(self, result: SplitEvalResult) -> Dict[str, int]:
        counts = {"true": 0, "false": 0, "none": 0}
        for question in result.questions:
            if question.parsed is True:
                counts["true"] += 1
            elif question.parsed is False:
                counts["false"] += 1
            else:
                counts["none"] += 1
        return counts

    def _append_epoch_metrics(
        self,
        epoch: int,
        train_loss: Optional[float],
        grad_norm: Optional[float],
        val_result: SplitEvalResult,
        empty_val_result: SplitEvalResult,
        checkpoint_path: Optional[str],
    ) -> None:
        mean_val_f1 = float(val_result.mean_f1 or 0.0)
        mean_val_em = float(val_result.mean_em or 0.0)
        mean_val_c_tokens = self._mean_c_tokens(val_result)
        empty_val_f1 = float(empty_val_result.mean_f1 or 0.0)
        empty_val_em = float(empty_val_result.mean_em or 0.0)
        for value, what in (
            (mean_val_f1, "mean_val_f1"),
            (mean_val_em, "mean_val_em"),
            (mean_val_c_tokens, "mean_val_c_tokens"),
            (empty_val_f1, "empty_val_f1"),
            (empty_val_em, "empty_val_em"),
        ):
            assert_finite_float(value, what)
        self.metrics.append(
            {
                "run_id": f"epoch-{epoch:05d}",
                "epoch": epoch,
                "mode": self.mode,
                "train_loss": train_loss,
                "grad_norm": grad_norm,
                "mean_val_f1": mean_val_f1,
                "mean_val_em": mean_val_em,
                "mean_val_c_tokens": mean_val_c_tokens,
                "empty_val_f1": empty_val_f1,
                "empty_val_em": empty_val_em,
                "best_val_f1": self.best_val_f1,
                "checkpoint_path": checkpoint_path,
            }
        )

    # -- step-0 baseline (C0) ------------------------------------------

    def _record_step0_baseline(self) -> Dict[str, Any]:
        """Evaluate the fresh C (== base model) before the first update."""
        baseline: Dict[str, Any] = {"epoch": 0}
        results: Dict[str, SplitEvalResult] = {}
        for split in ("val", "test"):
            with_reports = self._evaluate_split(split)
            empty = self._evaluate_split(split, empty_control=True)
            results[f"{split}"] = with_reports
            results[f"{split}_empty"] = empty
            baseline[f"{split}_with_reports"] = {
                "mean_f1": with_reports.mean_f1,
                "mean_em": with_reports.mean_em,
                "mean_c_tokens": self._mean_c_tokens(with_reports),
                "parsed": self._parsed_counts(with_reports),
            }
            baseline[f"{split}_empty_reports"] = {
                "mean_f1": empty.mean_f1,
                "mean_em": empty.mean_em,
                "mean_c_tokens": self._mean_c_tokens(empty),
                "parsed": self._parsed_counts(empty),
            }
        (self.stage_dir / "step0_baseline.json").write_text(
            json.dumps(baseline, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._append_epoch_metrics(
            epoch=0,
            train_loss=None,
            grad_norm=None,
            val_result=results["val"],
            empty_val_result=results["val_empty"],
            checkpoint_path=None,
        )
        logger.info(
            "step-0 baseline (C0): val F1=%.4f EM=%.4f empty F1=%.4f",
            baseline["val_with_reports"]["mean_f1"],
            baseline["val_with_reports"]["mean_em"],
            baseline["val_empty_reports"]["mean_f1"],
        )
        return baseline

    # -- final comparison ----------------------------------------------

    def _mean_ab_tokens(self, split: str) -> Dict[str, float]:
        cache = self.report_caches[split]
        entries = list(cache._data.values())
        if not entries:
            return {"mean_a_tokens": 0.0, "mean_b_tokens": 0.0,
                    "mean_ab_tokens": 0.0}
        a_tokens = [len(entry.get("a_ids", [])) for entry in entries]
        b_tokens = [len(entry.get("b_ids", [])) for entry in entries]
        return {
            "mean_a_tokens": sum(a_tokens) / len(a_tokens),
            "mean_b_tokens": sum(b_tokens) / len(b_tokens),
            "mean_ab_tokens": sum(a + b for a, b in zip(a_tokens, b_tokens))
            / len(entries),
        }

    def _condition_summary(self, result: SplitEvalResult) -> Dict[str, Any]:
        return {
            "mean_f1": result.mean_f1,
            "mean_em": result.mean_em,
            "mean_c_tokens": self._mean_c_tokens(result),
            "parsed": self._parsed_counts(result),
        }

    def _write_final_comparison(
        self,
        baseline: Dict[str, Any],
        c_phi_test: SplitEvalResult,
        c_phi_empty_test: SplitEvalResult,
    ) -> Dict[str, Any]:
        comparison = {
            "mode": self.mode,
            "test_report_cache": {
                "identity": self.report_cache_keys["test"],
                "path": str(self.report_caches["test"].cache_path),
                "num_questions": len(self.test_questions),
            },
            "test_ab_tokens": self._mean_ab_tokens("test"),
            "note": (
                "C0 was scored at step 0 before the first SFT update; "
                "C_phi after training, restored to the validation-best "
                "checkpoint; all four conditions share the identical "
                "cached test reports (A/B token counts come from the "
                "cache and are not counted as C's tokens)."
            ),
            "c0_with_reports": baseline["test_with_reports"],
            "c0_empty_reports": baseline["test_empty_reports"],
            "c_phi_with_reports": self._condition_summary(c_phi_test),
            "c_phi_empty_reports": self._condition_summary(c_phi_empty_test),
        }
        (self.stage_dir / "final_comparison.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return comparison

    def _write_report(self, comparison: Dict[str, Any]) -> None:
        lines = [
            "# Stage 2 SFT of C — report",
            "",
            f"- mode: {self.mode}",
            f"- worker checkpoint: {self.worker_checkpoint.adapter_dir}",
            f"  (best_step={self.worker_checkpoint.best_step}, "
            f"best_val_f1={self.worker_checkpoint.best_val_f1}, "
            f"resolution={self.worker_checkpoint.resolution})",
            f"- test report cache: {comparison['test_report_cache']['path']}",
            f"- questions: {comparison['test_report_cache']['num_questions']} "
            "(test)",
            "",
            "## Final test comparison (identical cached reports)",
            "",
            "| condition | mean F1 | mean EM | C tokens | parsed ok |",
            "|---|---|---|---|---|",
        ]
        for label, key in (
            ("C0 + reports", "c0_with_reports"),
            ("C_phi + reports", "c_phi_with_reports"),
            ("C0 + empty reports", "c0_empty_reports"),
            ("C_phi + empty reports", "c_phi_empty_reports"),
        ):
            summary = comparison[key]
            parsed = summary["parsed"]
            lines.append(
                f"| {label} | {summary['mean_f1']:.4f} | "
                f"{summary['mean_em']:.4f} | "
                f"{summary['mean_c_tokens']:.1f} | "
                f"{parsed['true']}/{parsed['true'] + parsed['false'] + parsed['none']} |"
            )
        ab = comparison["test_ab_tokens"]
        lines.append("")
        lines.append(
            "A/B tokens come from the cache (not counted as C's tokens): "
            f"mean A {ab['mean_a_tokens']:.1f}, mean B "
            f"{ab['mean_b_tokens']:.1f}."
        )
        (self.stage_dir / "report.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    # -- main loop -----------------------------------------------------

    def train(self) -> Dict[str, Any]:
        logger.info("Stage 2 SFT: %d epochs, batch size %d, lr %g",
                    self.cfg.sft.num_epochs, self.cfg.sft.batch_size,
                    self.cfg.sft.learning_rate)
        baseline = self._record_step0_baseline()

        for epoch in range(1, self.cfg.sft.num_epochs + 1):
            train_loss, grad_norm = self._train_epoch(epoch)
            val_result = self._evaluate_split("val")
            empty_val_result = self._evaluate_split("val", empty_control=True)
            improved = (
                val_result.mean_f1 is not None
                and val_result.mean_f1 > self.best_val_f1
            )
            if improved:
                self.best_val_f1 = float(val_result.mean_f1)
                self.best_epoch = epoch
                self._save_checkpoint(
                    "best_checkpoint", epoch, self.best_val_f1
                )
                logger.info(
                    "epoch %d: new best val F1 %.4f (checkpoint saved)",
                    epoch, self.best_val_f1,
                )
            epoch_dir = self._save_checkpoint(
                f"epoch_{epoch:03d}", epoch, self.best_val_f1
            )
            self._append_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                grad_norm=grad_norm,
                val_result=val_result,
                empty_val_result=empty_val_result,
                checkpoint_path=str(epoch_dir),
            )
            logger.info(
                "epoch %d: train_loss %.4f grad_norm %.4f val F1 %.4f "
                "EM %.4f (empty F1 %.4f)",
                epoch, train_loss, grad_norm,
                val_result.mean_f1 or 0.0, val_result.mean_em or 0.0,
                empty_val_result.mean_f1 or 0.0,
            )

        if self.best_epoch is None:
            raise RuntimeError(
                "no validation-best checkpoint was saved (mean_val_f1 "
                "never exceeded -1.0); refusing to continue"
            )
        # Restore the validation-best C, never the last epoch.
        self.synthesizer.load_adapter(
            self.stage_dir / "best_checkpoint" / "adapter"
        )
        logger.info(
            "restored validation-best C from epoch %d (val F1 %.4f)",
            self.best_epoch, self.best_val_f1,
        )

        c_phi_test = self._evaluate_split("test")
        c_phi_empty_test = self._evaluate_split("test", empty_control=True)
        comparison = self._write_final_comparison(
            baseline, c_phi_test, c_phi_empty_test
        )
        self._write_report(comparison)
        logger.info(
            "final test comparison: C0 F1 %.4f -> C_phi F1 %.4f "
            "(empty: %.4f -> %.4f)",
            comparison["c0_with_reports"]["mean_f1"],
            comparison["c_phi_with_reports"]["mean_f1"],
            comparison["c0_empty_reports"]["mean_f1"],
            comparison["c_phi_empty_reports"]["mean_f1"],
        )
        return {
            "mode": self.mode,
            "best_epoch": self.best_epoch,
            "best_val_f1": self.best_val_f1,
            "test_f1_c0": comparison["c0_with_reports"]["mean_f1"],
            "test_f1_c_phi": comparison["c_phi_with_reports"]["mean_f1"],
            "comparison": comparison,
        }
