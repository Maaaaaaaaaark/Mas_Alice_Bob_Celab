"""SFT trainer for Stage 1a worker SFT (``docs/WORKER_SFT.md``).

Teacher forcing only: each microbatch is ONE example whose full
assistant turn is supervised (never generation-then-alignment). The
loop per optimizer step is

    zero_grad -> forward -> per-example loss -> backward (scaled by the
    group size) -> grad clip -> optimizer.step -> scheduler.step

where the group size is the number of examples in this accumulation
group, computed BEFORE training from the deterministic example order,
so the final partial group keeps every example's weight exactly (the
loss never depends on where the group boundary falls).

The base model is verified frozen twice: a requires_grad audit at load
time and a parameter hash before/after training. Only LoRA parameters
are passed to the optimizer. Every effective setting (including weight
decay and the scheduler) is recorded in the run's config snapshot.

Trace mode records the full A-J evidence set for
``worked_example_worker_sft.md``; smoke/normal keep compact metrics.
A fake-model test is NOT a training result: ``real_model`` is recorded
from the model factory and shown in the worked example.
"""

from __future__ import annotations

import json
import logging
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from hotpot_mas.logging_io import JsonlWriter, collect_environment_info
from hotpot_mas.seeds import seed_all

from .config import WorkerSFTConfig
from .convert import WorkerExample
from .masking import (
    WorkerSFTExample,
    assert_supervised_bounds,
    build_worker_sft_example,
    per_example_loss,
    per_position_nll,
    supervised_token_records,
)
from .prompting import (
    build_messages,
    evaluate_output,
    load_instruction,
    shuffle_and_remap,
)
from .runner import load_prepared_examples
from .trace import WorkerSFTTraceRecorder, render_worker_sft_worked_example

logger = logging.getLogger("hotpot_mas.training.worker_sft.trainer")


def _git_revision() -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 - recorded as unavailable
        return "unavailable"


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"worker_sft-{stamp}"


def _example_shuffle_seed(example_id: str, order_seed: int) -> int:
    import hashlib

    digest = hashlib.sha256(
        f"{example_id}:{order_seed}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def _plan_groups(num_examples: int, grad_accum_steps: int) -> List[int]:
    """Group sizes for accumulation, computed BEFORE training.

    All groups except possibly the last have ``grad_accum_steps``
    examples; the last has the remainder. Group sizes never change mid
    run, so sample weights are identical wherever the boundary falls.
    """
    if num_examples <= 0:
        return []
    full = num_examples // grad_accum_steps
    remainder = num_examples % grad_accum_steps
    groups = [grad_accum_steps] * full
    if remainder:
        groups.append(remainder)
    return groups


class WorkerSFTTrainer:
    """Runs one worker-SFT training (trace / smoke / normal)."""

    def __init__(
        self,
        cfg: WorkerSFTConfig,
        mode: str,
        resume: bool = False,
        model_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.cfg = cfg
        self.mode = mode
        self.resume = resume
        self.model_factory = model_factory
        self.stage_dir = cfg.stage_dir()
        self.instruction = load_instruction(cfg.prompt_dir)
        self.metrics_writer: Optional[JsonlWriter] = None
        self._run_id: Optional[str] = None
        self._real_model = model_factory is None

    # ------------------------------------------------------------------
    # Run scaffolding
    # ------------------------------------------------------------------

    def _protect_fresh_run(self) -> None:
        artifacts = [
            "metrics.jsonl",
            "trace.json",
            "worked_example_worker_sft.md",
            "adapter",
        ]
        existing = [
            name for name in artifacts if (self.stage_dir / name).exists()
        ]
        if existing and not self.resume:
            raise FileExistsError(
                f"run artifacts already exist in {self.stage_dir}: "
                f"{existing}. Refusing to overwrite; pass --resume to "
                "continue an interrupted run."
            )

    def _load_model(self) -> Any:
        if self.model_factory is not None:
            model = self.model_factory()
            logger.info("using provided model factory (real_model=%s)",
                        self._real_model)
            return model
        from .model import HFLoraWorkerModel

        logger.info("loading real model %s ...", self.cfg.model.name)
        return HFLoraWorkerModel(self.cfg)

    def train(self) -> Dict[str, Any]:
        cfg = self.cfg
        self._protect_fresh_run()
        self.stage_dir.mkdir(parents=True, exist_ok=True)

        # Run identity.
        run_id = _run_id()
        self._run_id = run_id
        start_time = datetime.now(timezone.utc).isoformat()
        env_info = collect_environment_info()
        model = self._load_model()
        model_info = model.info()
        model_info["real_model"] = self._real_model
        seed_all(cfg.seed)

        prepared = load_prepared_examples(cfg)
        train_examples: List[WorkerExample] = prepared["prepared_train"]
        probe_examples: List[WorkerExample] = prepared["probes"]
        val_examples: List[WorkerExample] = prepared["prepared_val"]

        # Deterministic training order.
        order_seed = cfg.sft.example_order_seed
        train_order = sorted(train_examples, key=lambda e: e.example_id)
        rng = random.Random(order_seed)
        rng.shuffle(train_order)

        # Tokenized SFT examples (prompt + whole assistant turn).
        sft_examples: Dict[str, WorkerSFTExample] = {}
        length_stats: List[Dict[str, Any]] = []
        for example in train_order:
            messages, target = build_messages(
                example, self.instruction, _example_shuffle_seed(
                    example.example_id, order_seed
                )
            )
            sft_example = build_worker_sft_example(
                model.tokenizer, messages, target
            )
            assert_supervised_bounds(sft_example, cfg.model.max_input_tokens)
            sft_examples[example.example_id] = sft_example
            length_stats.append(
                {
                    "example_id": example.example_id,
                    "prompt_len": sft_example.prompt_len,
                    "total_len": int(sft_example.input_ids.shape[0]),
                }
            )

        # Fixed probe inputs (prompts only; same inputs before/after).
        fixed_inputs: List[Dict[str, Any]] = []
        for example in train_examples:
            messages, target = build_messages(
                example, self.instruction, _example_shuffle_seed(
                    example.example_id, order_seed
                )
            )
            fixed_inputs.append(
                {
                    "kind": "training",
                    "example": example,
                    "messages": messages,
                    "reference_target": target,
                    "shuffle_seed": _example_shuffle_seed(
                        example.example_id, order_seed
                    ),
                }
            )
        for example in probe_examples:
            messages, target = build_messages(example, self.instruction, 0)
            fixed_inputs.append(
                {
                    "kind": "probe",
                    "example": example,
                    "messages": messages,
                    "reference_target": target,
                    "shuffle_seed": 0,
                }
            )

        # Resume support comes FIRST: loading this run's own checkpoint
        # may replace the adapter modules (the HF wrapper rebuilds the
        # PeftModel), so the optimizer is constructed afterwards — an
        # optimizer built earlier would keep references to dead tensors
        # and every step would be a silent no-op.
        start_step, start_flat_index, resume_ckpt, saved_optimizer_state = (
            self._maybe_resume(model)
        )

        # Optimizer: AdamW over trainable (LoRA) parameters only.
        trainable = list(model.trainable_parameters())
        if not trainable:
            raise ValueError("model has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=cfg.sft.learning_rate,
            betas=tuple(cfg.sft.adam_betas),
            eps=cfg.sft.adam_eps,
            weight_decay=cfg.sft.weight_decay,
        )
        if saved_optimizer_state is not None:
            from ..policy import restore_optimizer_state

            restore_optimizer_state(optimizer, saved_optimizer_state)
        num_steps = len(train_order) * cfg.sft.num_epochs
        scheduler = self._build_scheduler(optimizer, num_steps)

        # Group plan (before training; never changes).
        group_sizes = []
        for _ in range(cfg.sft.num_epochs):
            group_sizes.extend(
                _plan_groups(len(train_order), cfg.sft.grad_accum_steps)
            )

        self.metrics_writer = JsonlWriter(self.stage_dir / "metrics.jsonl")

        # Pre-training generation on the fixed inputs (on a resumed run
        # this is the checkpointed model, so "before" means "before the
        # new optimizer steps", not "before any training ever").
        model.model.eval()
        pre_outputs = self._run_fixed_generations(model, fixed_inputs)

        # Frozen-base evidence (before).
        base_hash_before = model.base_param_hash(
            sample_every=None if self.mode == "trace" else 100
        )

        # ---- Training loop -------------------------------------------
        group_index = start_step
        flat_index = start_flat_index
        step = start_step
        per_example_nll_cache: Dict[str, Tensor] = {}
        per_step_records: List[Dict[str, Any]] = []
        group_losses: List[float] = []
        group_size = 0

        model.model.train()
        while group_index < len(group_sizes):
            size = group_sizes[group_index]
            optimizer.zero_grad()
            group_losses = []
            group_size = size
            for _ in range(size):
                example = train_order[flat_index % len(train_order)]
                flat_index += 1
                epoch = (flat_index - 1) // len(train_order)
                sft_example = sft_examples[example.example_id]
                input_ids = sft_example.input_ids.unsqueeze(0)
                labels = sft_example.labels.unsqueeze(0)
                logits = model.logits(
                    input_ids,
                    torch.ones_like(input_ids),
                )
                loss = per_example_loss(logits, labels).mean()
                loss = loss.to(torch.float32)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss for {example.example_id}"
                    )
                # Correct gradient-accumulation scaling: each example
                # contributes loss / group_size, so the accumulated
                # gradient equals the mean of the group's example
                # losses (per-example weighting preserved even for the
                # final partial group).
                (loss / group_size).backward()
                group_losses.append(float(loss.detach().cpu()))
                # Keep the actual forward's per-position NLLs for the
                # trace (last occurrence wins in multi-epoch runs).
                if self.mode == "trace":
                    with torch.no_grad():
                        nll_row = per_position_nll(logits, labels)
                    per_example_nll_cache[example.example_id] = (
                        nll_row[0].detach().cpu()
                    )

            grad_norm_before = self._grad_norm(model)
            torch.nn.utils.clip_grad_norm_(
                trainable, cfg.sft.max_grad_norm
            )
            grad_norm_after = self._grad_norm(model)
            if not all(math.isfinite(g) for g in
                       (grad_norm_before, grad_norm_after)):
                raise FloatingPointError("non-finite gradient norm")
            snapshot = (
                model.trainable_snapshot() if self.mode == "trace" else None
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            lora_delta = (
                model.trainable_delta_norm(snapshot)
                if snapshot is not None
                else None
            )
            update_loss = sum(group_losses) / group_size
            checkpoint = None
            if self.mode == "trace":
                checkpoint = self._save_checkpoint(
                    model, optimizer, step, flat_index
                )
            per_step_records.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "microsteps": group_size,
                    "learning_rate": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                    "update_loss": update_loss,
                    "per_example_losses": list(group_losses),
                    "grad_norm_before_clip": grad_norm_before,
                    "grad_norm_after_clip": grad_norm_after,
                    "lora_delta_norm": lora_delta,
                    "checkpoint": (
                        str(checkpoint) if checkpoint else None
                    ),
                }
            )
            self._write_metrics(per_step_records[-1])
            group_index += 1
            step += 1

        # Frozen-base evidence (after).
        base_hash_after = model.base_param_hash(
            sample_every=None if self.mode == "trace" else 100
        )
        base_unchanged = base_hash_before == base_hash_after

        # Post-training generation on the same fixed inputs.
        model.model.eval()
        post_outputs = self._run_fixed_generations(model, fixed_inputs)
        for item, pre, post in zip(fixed_inputs, pre_outputs, post_outputs):
            item["pre"] = pre
            item["post"] = post

        # Validation (smoke/normal only; trace defers model selection).
        val_records: List[Dict[str, Any]] = []
        if self.mode != "trace":
            val_records = self._evaluate_split(model, val_examples)
        else:
            val_records = [
                {
                    "note": (
                        "validation-based model selection is deferred; the "
                        "trace is a 10-step functional run"
                    )
                }
            ]

        # Checkpoints and outputs.
        final_ckpt = self._save_checkpoint(model, optimizer, step, flat_index)
        config_snapshot = cfg.to_dict()
        config_snapshot["_mode"] = self.mode
        config_snapshot["optimizer"] = {
            "name": type(optimizer).__name__,
            "lr": cfg.sft.learning_rate,
            "weight_decay": cfg.sft.weight_decay,
            "betas": cfg.sft.adam_betas,
            "eps": cfg.sft.adam_eps,
            "scheduler": cfg.sft.scheduler,
            "warmup_ratio": cfg.sft.warmup_ratio,
        }
        self._write_json(self.stage_dir / "config_snapshot.json", config_snapshot)
        self._write_json(
            self.stage_dir / "length_stats.json", {"examples": length_stats}
        )

        fixed_outputs = {
            "decode": cfg.decode.to_dict(),
            "inputs": [
                {
                    "kind": item["kind"],
                    "example_id": item["example"].example_id,
                    "reference_target": item["reference_target"],
                    "pre": item["pre"],
                    "post": item["post"],
                }
                for item in fixed_inputs
            ],
        }
        self._write_json(
            self.stage_dir / "fixed_example_outputs.json", fixed_outputs
        )

        # Trace + worked example (trace mode).
        trace_path = None
        md_path = None
        if self.mode == "trace":
            trace = self._build_trace(
                run_id=run_id,
                start_time=start_time,
                env_info=env_info,
                model_info=model_info,
                config_snapshot=config_snapshot,
                train_examples=train_examples,
                train_order=train_order,
                probe_examples=probe_examples,
                fixed_inputs=fixed_inputs,
                pre_outputs=pre_outputs,
                post_outputs=post_outputs,
                sft_examples=sft_examples,
                per_example_nll_cache=per_example_nll_cache,
                per_step_records=per_step_records,
                base_hash_before=base_hash_before,
                base_hash_after=base_hash_after,
                base_unchanged=base_unchanged,
                val_records=val_records,
                dedup_report=self._load_dedup_report(),
                label_quality=self._load_label_quality(),
                model=model,
            )
            trace_path = self.stage_dir / "trace.json"
            self._write_json(trace_path, trace)
            md_path = self.stage_dir / "worked_example_worker_sft.md"
            render_worker_sft_worked_example(trace_path, md_path)

        return {
            "mode": self.mode,
            "run_id": run_id,
            "stage_dir": str(self.stage_dir),
            "train_examples": len(train_examples),
            "optimizer_steps": step - start_step,
            "final_checkpoint": str(final_ckpt),
            "base_weights_unchanged": base_unchanged,
            "trace": str(trace_path) if trace_path else None,
            "worked_example": str(md_path) if md_path else None,
            "val_mean_correct": (
                sum(r["correct"] for r in val_records if "correct" in r)
                / max(
                    len([r for r in val_records if "correct" in r]), 1
                )
                if val_records and "correct" in val_records[0]
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Loop helpers
    # ------------------------------------------------------------------

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, num_steps: int
    ) -> Optional[Any]:
        cfg = self.cfg.sft
        if cfg.scheduler == "none":
            return None
        if cfg.scheduler == "linear":
            warmup = max(1, int(cfg.warmup_ratio * num_steps))
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.0,
                end_factor=1.0,
                total_iters=warmup,
            )
        raise ValueError(f"unknown scheduler {cfg.scheduler!r}")

    @staticmethod
    def _grad_norm(model: Any) -> float:
        total = 0.0
        for param in model.trainable_parameters():
            if param.grad is not None:
                total += float(param.grad.detach().float().norm().item()) ** 2
        return total**0.5

    def _write_metrics(self, record: Dict[str, Any]) -> None:
        if self.metrics_writer is None:
            raise RuntimeError("metrics writer not initialized")
        if self._run_id is None:
            raise RuntimeError("run id not set")
        # One writer row per optimizer step; step-scoped ids keep the
        # append-only writer's duplicate detection meaningful on resume.
        self.metrics_writer.append(
            {**record, "run_id": f"{self._run_id}-step{record['step']:05d}"}
        )

    def _maybe_resume(
        self, model: Any
    ) -> Tuple[int, int, Optional[Path], Optional[Dict[str, Any]]]:
        """Load this run's own checkpoint adapter (resume).

        Returns ``(start_step, start_flat_index, adapter_dir,
        saved_optimizer_state)``. The adapter is loaded here, but the
        optimizer state is returned to the caller instead of restored,
        because the optimizer must be BUILT after the adapter load (the
        HF wrapper replaces the adapter modules entirely).
        """
        if not self.resume:
            return 0, 0, None, None
        # Checkpoints live in step_*/state.json; resume from the latest
        # step so a preempted run continues exactly where it stopped.
        step_dirs = sorted(
            (p for p in self.stage_dir.glob("step_*") if p.is_dir()),
            key=lambda p: int(p.name.split("_")[1]),
        )
        if not step_dirs:
            raise FileNotFoundError(
                "--resume requested but no step checkpoints in "
                f"{self.stage_dir}"
            )
        state_path = step_dirs[-1] / "state.json"
        if not state_path.is_file():
            raise FileNotFoundError(
                f"resume checkpoint has no state.json: {state_path}"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        adapter_dir = step_dirs[-1] / "adapter"
        if not adapter_dir.is_dir():
            raise FileNotFoundError(
                f"resume adapter missing: {adapter_dir}"
            )
        model.load_adapter(adapter_dir)
        return (
            int(state["step"]),
            int(state.get("flat_index", 0)),
            adapter_dir,
            state.get("optimizer_state"),
        )

    def _save_checkpoint(
        self, model: Any, optimizer: torch.optim.Optimizer, step: int,
        flat_index: int,
    ) -> Path:
        from ..policy import remap_optimizer_state

        step_dir = self.stage_dir / f"step_{step:05d}"
        adapter_dir = step_dir / "adapter"
        model.save_adapter(adapter_dir)
        state = {
            "step": step,
            "flat_index": flat_index,
            "experiment_version": self.cfg.experiment_version,
            "mode": self.mode,
            "run_config": self.cfg.to_dict(),
            "optimizer_state": remap_optimizer_state(
                optimizer, [name for name, _ in model.model.named_parameters()
                            if _.requires_grad]
            ),
        }
        import json as _json

        (step_dir / "state.json").write_text(
            _json.dumps(state, indent=2), encoding="utf-8"
        )
        return adapter_dir

    # ------------------------------------------------------------------
    # Generation + evaluation helpers
    # ------------------------------------------------------------------

    def _run_fixed_generations(
        self, model: Any, fixed_inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        outputs = []
        for item in fixed_inputs:
            generated = model.generate(
                item["messages"],
                max_new_tokens=self.cfg.decode.max_new_tokens,
                seed=self.cfg.decode.probe_seed,
            )
            ordered, _ = shuffle_and_remap(item["example"], item["shuffle_seed"])
            checks = evaluate_output(
                item["example"], ordered, generated["text"]
            )
            outputs.append(
                {
                    "text": generated["text"],
                    "num_new_tokens": generated["num_new_tokens"],
                    "finish_reason": generated["finish_reason"],
                    "checks": checks,
                }
            )
        return outputs

    def _evaluate_split(
        self, model: Any, examples: List[WorkerExample]
    ) -> List[Dict[str, Any]]:
        records = []
        for example in examples:
            messages, _ = build_messages(example, self.instruction, 0)
            generated = model.generate(
                messages,
                max_new_tokens=self.cfg.decode.max_new_tokens,
                seed=self.cfg.decode.probe_seed,
            )
            ordered, _ = shuffle_and_remap(example, 0)
            checks = evaluate_output(example, ordered, generated["text"])
            records.append(
                {
                    "example_id": example.example_id,
                    "output": generated["text"],
                    "correct": checks["correct"],
                    "checks": checks,
                }
            )
        return records

    # ------------------------------------------------------------------
    # Trace assembly
    # ------------------------------------------------------------------

    def _load_dedup_report(self) -> Dict[str, Any]:
        path = self.cfg.prepared_data_dir() / "dedup_report.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_label_quality(self) -> Dict[str, Any]:
        path = self.cfg.prepared_data_dir() / "label_quality_report.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")

    def _build_trace(self, **kwargs: Any) -> Dict[str, Any]:
        recorder = WorkerSFTTraceRecorder(self.cfg, self.mode)
        return recorder.build(**kwargs)


def run_worker_sft_training(
    cfg: WorkerSFTConfig,
    mode: str,
    resume: bool = False,
    model_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Convenience entry point used by the CLI."""
    trainer = WorkerSFTTrainer(
        cfg, mode, resume=resume, model_factory=model_factory
    )
    return trainer.train()
