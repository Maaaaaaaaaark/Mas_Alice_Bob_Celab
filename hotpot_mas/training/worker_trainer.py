"""Stage 1: cross-paired GRPO training of the shared worker LoRA
(Algorithm 1 of ``cross_paired_grpo.tex``).

Loop outline per update:

1. Collect ``N`` signal questions: rollout each candidate question
   (G reports per worker under the current policy, G x G frozen-C scores,
   cross-paired marginals), keep the questions where at least one side
   signals (``std(Q_side) > delta``), bounded by ``max_sampling_attempts``.
2. Teacher-force every kept report under the current policy and run
   ``num_policy_epochs`` optimizer steps on ``L = -J`` (ratio-clipped
   objective over report tokens only, no KL penalty). The gradient
   accumulates over minibatches of questions so J keeps the exact TeX
   weighting ``1/(|Q| |S_q|)`` per report.
3. Periodically evaluate the workers deterministically on the validation
   split (cached per checkpoint step) and keep the best checkpoint by
   validation F1; write one ``metrics.jsonl`` line per update.

The workers A and B share **one** ``Policy`` instance (one base model, one
LoRA adapter); they differ only in role prompt and private evidence. The
synthesizer C is a separate frozen base-model instance with no trainable
parameters and is called exactly ``G * G`` times per question.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from hotpot_mas.evaluation import normalize_answer
from hotpot_mas.logging_io import JsonlWriter, collect_environment_info
from hotpot_mas.seeds import seed_all

from .config import TrainingConfig
from .data import describe_split, load_questions_for_split
from .eval import (
    EvalReportCache,
    SplitEvalResult,
    cache_identity,
    evaluate_workers,
)
from .grpo_loss import (
    approx_kl_per_report,
    batch_loss,
    clip_fraction_per_report,
    entropy_per_report,
    per_report_objectives,
)
from .policy import HFPolicy, Policy, remap_optimizer_state, restore_optimizer_state
from .prompts_builder import TrainingPrompts
from .rollout import QuestionRollout, RolloutReport, rollout_question
from .synthesizers import HFSynthesizer, Synthesizer
from .trace import TraceRecorder, render_worked_example, write_trace_json

logger = logging.getLogger(__name__)


@dataclass
class KeptReportEntry:
    """One kept (signal-side) report plus everything the loss needs."""

    rollout_report: RolloutReport
    old_logp: Tensor  # [T], detached rollout log probs
    mask: Tensor  # [T] float ones (report tokens only)
    advantage: float
    weight: float  # 1 / (|Q| * |S_q|)


@dataclass
class EpochRecord:
    """Detached teacher-forcing values of one report in one epoch."""

    entry: KeptReportEntry
    new_logp: Tensor  # [T] detached


@dataclass
class UpdateOutcome:
    """Everything recorded about one completed update."""

    step: int
    attempted: int
    collected: int
    discarded: int
    signal_counts: Dict[str, int]
    mean_reward: float
    loss: float
    grad_norm: float
    approx_kl: float
    clip_fraction: float
    entropy: float
    mean_a_tokens: float
    mean_b_tokens: float
    mean_c_tokens: float
    checkpoint_path: str
    insufficient_signal: bool = False
    skipped: bool = False
    epoch_records: List[EpochRecord] = field(default_factory=list)
    rollouts: List[QuestionRollout] = field(default_factory=list)
    val_result: Optional[SplitEvalResult] = None
    param_before: Optional[Dict[str, Any]] = None
    param_after: Optional[Dict[str, Any]] = None


def _chunk_questions(
    entries_by_question: List[List[KeptReportEntry]], chunk_size: int
) -> List[List[KeptReportEntry]]:
    """Group kept reports into question-chunks of ``chunk_size`` questions."""
    chunks: List[List[KeptReportEntry]] = []
    current: List[KeptReportEntry] = []
    count = 0
    for question_entries in entries_by_question:
        current.extend(question_entries)
        count += 1
        if count >= chunk_size:
            chunks.append(current)
            current = []
            count = 0
    if current:
        chunks.append(current)
    return chunks


class WorkerTrainer:
    """Stage 1 GRPO trainer; policy/synthesizer injectable for tests."""

    def __init__(
        self,
        cfg: TrainingConfig,
        mode: str = "smoke",
        resume: bool = False,
        policy_factory: Optional[Callable[[], Policy]] = None,
        synthesizer_factory: Optional[Callable[[], Synthesizer]] = None,
    ):
        self.cfg = cfg
        self.mode = mode
        self.stage_dir = cfg.stage_dir()
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.environment = collect_environment_info()

        seed_all(cfg.seed)
        self.prompts = TrainingPrompts(cfg.prompt_dir)
        self.policy = (
            policy_factory() if policy_factory else HFPolicy(cfg.model)
        )
        self.synthesizer = (
            synthesizer_factory()
            if synthesizer_factory
            else HFSynthesizer(cfg.model, cfg.workers.synthesizer_decode)
        )
        params = self.policy.trainable_parameters()
        if not params:
            raise RuntimeError("policy has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg.workers.learning_rate,
            weight_decay=cfg.workers.weight_decay,
        )

        self.metrics_path = self.stage_dir / "metrics.jsonl"
        self.metrics = JsonlWriter(self.metrics_path)

        self.train_questions = load_questions_for_split(cfg, "train")
        self.val_questions = load_questions_for_split(cfg, "val")
        self.test_questions = load_questions_for_split(cfg, "test")

        self.start_step = 0
        self.best_val_f1 = -1.0
        self.best_step: Optional[int] = None
        self.best_policy_state: Optional[Dict[str, Tensor]] = None
        self.q_pointer = 0
        if resume:
            self._load_latest_checkpoint()

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    def _param_names(self) -> List[str]:
        return list(self.policy.state_dict().keys())

    def _save_checkpoint(
        self, step: int, best_val_f1: float
    ) -> Path:
        ckpt_dir = self.stage_dir / f"step_{step:05d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save_adapter(ckpt_dir / "adapter")
        torch.save(
            remap_optimizer_state(self.optimizer, self._param_names()),
            ckpt_dir / "optimizer.pt",
        )
        torch.save(
            {
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "torch_cuda_rng": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            ckpt_dir / "rng.pt",
        )
        (ckpt_dir / "state.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "best_val_f1": best_val_f1,
                    "best_step": self.best_step,
                    "q_pointer": self.q_pointer,
                    "mode": self.mode,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return ckpt_dir

    @staticmethod
    def _torch_load(path: Path) -> Any:
        """Load a trusted checkpoint written by this trainer.

        PyTorch 2.6 changed ``torch.load`` to ``weights_only=True`` by
        default, which cannot deserialize NumPy/Python RNG state.
        """
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # compatibility with older PyTorch
            return torch.load(path, map_location="cpu")

    def _load_latest_checkpoint(self) -> None:
        step_dirs = sorted(self.stage_dir.glob("step_*"))
        if not step_dirs:
            raise RuntimeError(
                f"--resume requested but no checkpoints found in {self.stage_dir}"
            )
        latest = step_dirs[-1]
        state = json.loads((latest / "state.json").read_text(encoding="utf-8"))
        self.policy.load_adapter(latest / "adapter")
        saved = self._torch_load(latest / "optimizer.pt")
        restore_optimizer_state(self.optimizer, saved)
        rng = self._torch_load(latest / "rng.pt")
        random.setstate(rng["python_rng"])
        np.random.set_state(rng["numpy_rng"])
        torch.set_rng_state(rng["torch_rng"])
        if rng.get("torch_cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda_rng"])
        self.start_step = int(state["step"])
        self.best_val_f1 = float(state["best_val_f1"])
        raw_best_step = state.get("best_step")
        self.best_step = (
            int(raw_best_step) if raw_best_step is not None else None
        )
        best_state_path = self.stage_dir / "best_policy_state.pt"
        if best_state_path.is_file():
            self.best_policy_state = self._torch_load(best_state_path)
        self.q_pointer = int(state.get("q_pointer", 0))
        logger.info(
            "resumed from %s (step %d, best_val_f1 %.4f)",
            latest, self.start_step, self.best_val_f1,
        )

    # ------------------------------------------------------------------
    # rollout collection
    # ------------------------------------------------------------------

    def _collect_signal_questions(
        self,
    ) -> Tuple[List[QuestionRollout], int, int, Dict[str, int]]:
        """Collect up to N signal questions (bounded attempts)."""
        workers = self.cfg.workers
        rollouts: List[QuestionRollout] = []
        attempts = 0
        discarded = 0
        signal_counts = {"A": 0, "B": 0, "both": 0}
        while (
            len(rollouts) < workers.questions_per_update
            and attempts < workers.max_sampling_attempts
        ):
            question = self.train_questions[
                self.q_pointer % len(self.train_questions)
            ]
            self.q_pointer += 1
            attempts += 1
            rollout = rollout_question(
                question,
                self.policy,
                self.synthesizer,
                self.prompts,
                workers,
                self.cfg.seed,
                rollout_index=self.q_pointer - 1,
            )
            if rollout.has_signal:
                rollouts.append(rollout)
                if "A" in rollout.signal.kept_sides and "B" in rollout.signal.kept_sides:
                    signal_counts["both"] += 1
                elif "A" in rollout.signal.kept_sides:
                    signal_counts["A"] += 1
                elif "B" in rollout.signal.kept_sides:
                    signal_counts["B"] += 1
            else:
                discarded += 1
        return rollouts, attempts, discarded, signal_counts

    # ------------------------------------------------------------------
    # GRPO update
    # ------------------------------------------------------------------

    def _build_entries(
        self, rollouts: List[QuestionRollout]
    ) -> List[List[KeptReportEntry]]:
        entries_by_question: List[List[KeptReportEntry]] = []
        for rollout in rollouts:
            question_entries: List[KeptReportEntry] = []
            for rr in rollout.kept_reports:
                if rr.advantage is None:
                    raise RuntimeError("kept report without advantage")
                question_entries.append(
                    KeptReportEntry(
                        rollout_report=rr,
                        old_logp=torch.tensor(rr.report.logprobs),
                        mask=torch.ones(len(rr.report.token_ids)),
                        advantage=rr.advantage,
                        weight=1.0
                        / (len(rollouts) * len(rollout.kept_reports)),
                    )
                )
            entries_by_question.append(question_entries)
        return entries_by_question

    def _run_epoch(
        self, entries_by_question: List[List[KeptReportEntry]]
    ) -> Tuple[List[EpochRecord], float, float]:
        """One policy epoch: accumulate L = -J over question chunks, step."""
        workers = self.cfg.workers
        chunks = _chunk_questions(entries_by_question, workers.minibatch_size)
        self.optimizer.zero_grad()
        records: List[EpochRecord] = []
        loss_value = 0.0
        for chunk in chunks:
            new_logps: List[Tensor] = []
            old_logps: List[Tensor] = []
            advantages: List[float] = []
            masks: List[Tensor] = []
            weights: List[float] = []
            for entry in chunk:
                new_logp = self.policy.teacher_force(
                    entry.rollout_report.prompt_ids,
                    entry.rollout_report.report.token_ids,
                    workers.rollout,
                )
                new_logps.append(new_logp)
                old_logps.append(entry.old_logp)
                advantages.append(entry.advantage)
                masks.append(entry.mask)
                weights.append(entry.weight)
                records.append(
                    EpochRecord(entry=entry, new_logp=new_logp.detach())
                )
            device = new_logps[0].device
            dtype = new_logps[0].dtype
            lengths = torch.tensor(
                [value.numel() for value in new_logps], device=device
            )
            stacked_new = pad_sequence(
                new_logps, batch_first=True, padding_value=0.0
            )
            stacked_old = pad_sequence(
                [value.to(device=device, dtype=dtype) for value in old_logps],
                batch_first=True,
                padding_value=0.0,
            )
            positions = torch.arange(
                stacked_new.shape[1], device=device
            ).unsqueeze(0)
            mask_t = (positions < lengths.unsqueeze(1)).to(dtype=dtype)
            adv_t = torch.tensor(advantages, device=device, dtype=dtype)
            weight_t = torch.tensor(weights, device=device, dtype=dtype)
            objectives = per_report_objectives(
                stacked_new,
                stacked_old,
                adv_t,
                mask_t,
                workers.clip_epsilon,
            )
            loss = batch_loss(objectives, weight_t)
            loss.backward()
            loss_value += float(loss.detach().item())
        grad_norm = self._grad_norm()
        self.optimizer.step()
        return records, loss_value, grad_norm

    def _grad_norm(self) -> float:
        total = 0.0
        for param in self.policy.trainable_parameters():
            if param.grad is not None:
                total += float(param.grad.detach().float().norm().item()) ** 2
        return total**0.5

    def _epoch_diagnostics(
        self, records: List[EpochRecord], clip_epsilon: float
    ) -> Tuple[float, float, float]:
        """Mean approx KL, clip fraction, and entropy over the records."""
        if not records:
            return 0.0, 0.0, 0.0
        device = records[0].new_logp.device
        dtype = records[0].new_logp.dtype
        new_values = [r.new_logp for r in records]
        old_values = [
            r.entry.old_logp.to(device=device, dtype=dtype) for r in records
        ]
        lengths = torch.tensor(
            [value.numel() for value in new_values], device=device
        )
        new_stack = pad_sequence(
            new_values, batch_first=True, padding_value=0.0
        )
        old_stack = pad_sequence(
            old_values, batch_first=True, padding_value=0.0
        )
        positions = torch.arange(
            new_stack.shape[1], device=device
        ).unsqueeze(0)
        mask_stack = (positions < lengths.unsqueeze(1)).to(dtype=dtype)
        kl = float(
            approx_kl_per_report(new_stack, old_stack, mask_stack).mean().item()
        )
        cf = float(
            clip_fraction_per_report(
                new_stack, old_stack, mask_stack, clip_epsilon
            ).mean().item()
        )
        ent = float(
            entropy_per_report(new_stack, mask_stack).mean().item()
        )
        return kl, cf, ent

    def _apply_grpo_update(
        self,
        step: int,
        rollouts: List[QuestionRollout],
        attempted: int,
        discarded: int,
        signal_counts: Dict[str, int],
    ) -> UpdateOutcome:
        workers = self.cfg.workers
        entries_by_question = self._build_entries(rollouts)
        param_before = self.policy.parameter_norm_summary()
        epoch_records: List[EpochRecord] = []
        loss_value = 0.0
        grad_norm = 0.0
        for _ in range(workers.num_policy_epochs):
            records, loss_value, grad_norm = self._run_epoch(
                entries_by_question
            )
            epoch_records = records  # keep the last epoch's values
        param_after = self.policy.parameter_norm_summary()
        kl, cf, ent = self._epoch_diagnostics(
            epoch_records, workers.clip_epsilon
        )

        pair_f1s = [
            pair.synth.f1
            for rollout in rollouts
            for row in rollout.pairs
            for pair in row
        ]
        mean_reward = sum(pair_f1s) / len(pair_f1s) if pair_f1s else 0.0

        mean_a_tokens = (
            sum(rr.report.num_tokens for rollout in rollouts for rr in rollout.a_reports)
            / len(rollouts)
            / max(workers.G, 1)
        )
        mean_b_tokens = (
            sum(rr.report.num_tokens for rollout in rollouts for rr in rollout.b_reports)
            / len(rollouts)
            / max(workers.G, 1)
        )
        pair_outputs = [
            pair.synth.generated_tokens
            for rollout in rollouts
            for row in rollout.pairs
            for pair in row
        ]
        mean_c_tokens = (
            sum(pair_outputs) / len(pair_outputs) if pair_outputs else 0.0
        )

        return UpdateOutcome(
            step=step,
            attempted=attempted,
            collected=len(rollouts),
            discarded=discarded,
            signal_counts=signal_counts,
            mean_reward=mean_reward,
            loss=loss_value,
            grad_norm=grad_norm,
            approx_kl=kl,
            clip_fraction=cf,
            entropy=ent,
            mean_a_tokens=mean_a_tokens,
            mean_b_tokens=mean_b_tokens,
            mean_c_tokens=mean_c_tokens,
            checkpoint_path="",
            insufficient_signal=len(rollouts) < workers.questions_per_update,
            epoch_records=epoch_records,
            rollouts=rollouts,
            param_before=param_before,
            param_after=param_after,
        )

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, split: str, step: int) -> SplitEvalResult:
        questions = {
            "val": self.val_questions,
            "test": self.test_questions,
        }[split]
        key = cache_identity(
            f"step_{step:05d}",
            split,
            self.cfg.workers.eval_decode,
            self.prompts,
            self.cfg.prompt_dir,
        )
        cache = EvalReportCache(
            self.stage_dir / "eval_cache" / f"{key}.json"
        )
        return evaluate_workers(
            self.policy,
            self.synthesizer,
            questions,
            self.prompts,
            self.cfg.workers,
            split,
            self.cfg.seed,
            cache,
        )

    # ------------------------------------------------------------------
    # metrics + trace
    # ------------------------------------------------------------------

    def _write_metrics(
        self,
        outcome: UpdateOutcome,
        trace_mode: bool,
    ) -> None:
        workers = self.cfg.workers
        val_f1 = None
        val_em = None
        val_note = None
        if outcome.val_result is not None and outcome.val_result.mean_f1 is not None:
            val_f1 = outcome.val_result.mean_f1
            val_em = outcome.val_result.mean_em
        elif trace_mode:
            val_note = "validation evaluation skipped in trace mode"
        self.metrics.append(
            {
                "run_id": f"update-{outcome.step:05d}",
                "update": outcome.step,
                "mode": self.mode,
                "attempted_questions": outcome.attempted,
                "signal_questions": outcome.collected,
                "discarded_questions": outcome.discarded,
                "discard_rate": (
                    outcome.discarded / outcome.attempted
                    if outcome.attempted
                    else 0.0
                ),
                "signal_rate_a": (
                    outcome.signal_counts["A"] / outcome.collected
                    if outcome.collected
                    else 0.0
                ),
                "signal_rate_b": (
                    outcome.signal_counts["B"] / outcome.collected
                    if outcome.collected
                    else 0.0
                ),
                "signal_rate_both": (
                    outcome.signal_counts["both"] / outcome.collected
                    if outcome.collected
                    else 0.0
                ),
                "insufficient_signal": outcome.insufficient_signal,
                "mean_reward": outcome.mean_reward,
                "mean_val_f1": val_f1,
                "mean_val_em": val_em,
                "val_note": val_note,
                "mean_a_tokens": outcome.mean_a_tokens,
                "mean_b_tokens": outcome.mean_b_tokens,
                "mean_c_tokens": outcome.mean_c_tokens,
                "loss": outcome.loss,
                "grad_norm": outcome.grad_norm,
                "approx_kl": outcome.approx_kl,
                "clip_fraction": outcome.clip_fraction,
                "entropy": outcome.entropy,
                "num_policy_epochs": workers.num_policy_epochs,
                "best_val_f1": self.best_val_f1,
                "checkpoint_path": outcome.checkpoint_path,
            }
        )

    # ------------------------------------------------------------------
    # trace assembly (trace mode only)
    # ------------------------------------------------------------------

    def _assemble_trace(
        self, outcome: UpdateOutcome
    ) -> Dict[str, Any]:
        cfg = self.cfg
        rollout = outcome.rollouts[0]
        question = rollout.question
        workers = cfg.workers
        env = self.environment

        documents = [
            {
                "title": doc["title"],
                "paragraph": doc["paragraph"],
                "is_supporting": bool(doc["is_supporting"]),
            }
            for doc in question.document_pool
        ]
        meta = question.partition_metadata

        reports: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": []}
        for side, side_reports in (("A", rollout.a_reports), ("B", rollout.b_reports)):
            for rr in side_reports:
                reports[side].append(
                    {
                        "index": rr.index,
                        "token_ids": rr.report.token_ids,
                        "text": rr.report.text,
                        "num_tokens": rr.report.num_tokens,
                        "logprobs": rr.report.logprobs,
                        "generation_seed": rr.generation_seed,
                        "finish_reason": rr.report.finish_reason,
                    }
                )

        pairs: List[List[Dict[str, Any]]] = []
        for row in rollout.pairs:
            row_out: List[Dict[str, Any]] = []
            for pair in row:
                row_out.append(
                    {
                        "i": pair.i,
                        "j": pair.j,
                        "messages": pair.messages,
                        "raw_output": pair.synth.raw_output,
                        "pred_answer": pair.synth.pred_answer,
                        "parsed": pair.synth.parsed,
                        "normalized_pred": normalize_answer(
                            pair.synth.pred_answer
                        ),
                        "normalized_gold": normalize_answer(question.answer),
                        "precision": pair.synth.precision,
                        "recall": pair.synth.recall,
                        "f1": pair.synth.f1,
                        "input_tokens": pair.synth.input_tokens,
                        "generated_tokens": pair.synth.generated_tokens,
                        "finish_reason": pair.synth.finish_reason,
                    }
                )
            pairs.append(row_out)

        signal = rollout.signal
        loss_entries: List[Dict[str, Any]] = []
        for record in outcome.epoch_records:
            entry = record.entry
            old_logp = entry.old_logp.tolist()
            new_logp = record.new_logp.tolist()
            ratio = [
                float(np.exp(n - o)) for n, o in zip(new_logp, old_logp)
            ]
            # Recompute the objective of this report under the same code
            # path used for training (detached, for the record only).
            objective_tensor = per_report_objectives(
                record.new_logp.unsqueeze(0),
                entry.old_logp.to(record.new_logp.device).unsqueeze(0),
                torch.tensor(
                    [entry.advantage], device=record.new_logp.device
                ),
                entry.mask.to(record.new_logp.device).unsqueeze(0),
                workers.clip_epsilon,
            )
            loss_entries.append(
                {
                    "side": entry.rollout_report.side,
                    "index": entry.rollout_report.index,
                    "tokens": len(old_logp),
                    "weight": entry.weight,
                    "objective": float(objective_tensor.item()),
                    "new_logp": new_logp,
                    "old_logp": old_logp,
                    "ratio": ratio,
                }
            )
        J = sum(e["weight"] * e["objective"] for e in loss_entries)

        rec = TraceRecorder(
            run={
                "mode": self.mode,
                "config_summary": cfg.to_dict(),
                "command": "train-workers",
            },
            dataset={
                "dataset": cfg.data.dataset,
                "dataset_config": cfg.data.dataset_config,
                "split": cfg.data.train.split,
                "question_id": question.question_id,
                "seed": cfg.seed,
                "partition_note": (
                    "oracle-balanced partition using gold supporting-document "
                    "metadata; not an official HotpotQA agent split"
                ),
            },
            question={"question": question.question, "gold": question.answer},
            documents=documents,
            partition={
                "rule": meta.get("rule"),
                "evidence_alice": question.evidence_alice,
                "evidence_bob": question.evidence_bob,
                "alice_documents": meta.get("alice_documents"),
                "bob_documents": meta.get("bob_documents"),
                "single_reader_documents": meta.get("single_reader_documents"),
            },
            prompts={
                "worker_a_messages": rollout.a_reports[0].messages,
                "worker_b_messages": rollout.b_reports[0].messages,
                "prompt_ids_length_a": len(rollout.a_reports[0].prompt_ids),
                "prompt_ids_length_b": len(rollout.b_reports[0].prompt_ids),
                "decode_params": workers.rollout.to_dict(),
                "prompt_version": self.prompts.version,
                "prompt_hashes": self.prompts.hashes,
            },
            reports=reports,
            pairs=pairs,
            reward_matrix=rollout.reward_matrix(),
            marginals={
                "q_a": signal.q_a,
                "q_b": signal.q_b,
                "std_a": signal.std_a,
                "std_b": signal.std_b,
                "signal_a": signal.signal_a,
                "signal_b": signal.signal_b,
            },
            advantages={
                "a": signal.advantages_a if signal.advantages_a is not None else [],
                "b": signal.advantages_b if signal.advantages_b is not None else [],
            },
            kept={
                "sides": signal.kept_sides,
                "reports": [
                    {"side": rr.side, "index": rr.index}
                    for rr in rollout.kept_reports
                ],
                "excluded_empty_reports": rollout.excluded_empty_reports,
            },
            loss={"per_report": loss_entries, "J": J, "L": -J},
            diagnostics={
                "approx_kl": outcome.approx_kl,
                "clip_fraction": outcome.clip_fraction,
                "entropy": outcome.entropy,
                "grad_norm": outcome.grad_norm,
                "learning_rate": workers.learning_rate,
                "num_policy_epochs": workers.num_policy_epochs,
                "delta": workers.delta,
                "eps_n": workers.eps_n,
                "clip_epsilon": workers.clip_epsilon,
                "G": workers.G,
            },
            parameters={
                "before": outcome.param_before,
                "after": outcome.param_after,
            },
            checkpoint={
                "path": outcome.checkpoint_path,
                "saved": True,
            },
            environment=env,
        )
        return rec.to_dict()

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        cfg = self.cfg
        workers = cfg.workers
        trace_mode = self.mode == "trace"
        logger.info(describe_split(cfg))
        print(describe_split(cfg))

        final_outcomes: List[UpdateOutcome] = []
        for step in range(self.start_step + 1, workers.steps + 1):
            rollouts, attempted, discarded, signal_counts = (
                self._collect_signal_questions()
            )
            if not rollouts:
                if workers.fail_on_insufficient_signal:
                    raise RuntimeError(
                        f"no signal questions collected in {attempted} "
                        f"attempts at update {step} (fail_on_insufficient_signal)"
                    )
                logger.warning(
                    "update %d skipped: no signal questions in %d attempts",
                    step, attempted,
                )
                continue
            outcome = self._apply_grpo_update(
                step, rollouts, attempted, discarded, signal_counts
            )
            final_outcomes.append(outcome)

            # Periodic (and final) validation evaluation, before the
            # checkpoint is written so state.json records the best val F1
            # known at this step.
            if not trace_mode and (
                step % workers.eval_interval == 0 or step == workers.steps
            ):
                val_result = self._evaluate("val", step)
                outcome.val_result = val_result
                if (
                    val_result.mean_f1 is not None
                    and val_result.mean_f1 > self.best_val_f1
                ):
                    self.best_val_f1 = val_result.mean_f1
                    self.best_step = step
                    self.best_policy_state = self.policy.state_dict()
                    torch.save(
                        self.best_policy_state,
                        self.stage_dir / "best_policy_state.pt",
                    )
                    logger.info(
                        "update %d: new best val F1 %.4f",
                        step, self.best_val_f1,
                    )

            outcome.checkpoint_path = str(
                self._save_checkpoint(step, self.best_val_f1)
            )

            if trace_mode:
                trace_data = self._assemble_trace(outcome)
                trace_path = self.stage_dir / "trace.json"
                write_trace_json(trace_path, trace_data)
                worked_path = render_worked_example(
                    trace_path, self.stage_dir / "worked_example.md"
                )
                logger.info(
                    "trace written: %s and %s", trace_path, worked_path
                )

            self._write_metrics(outcome, trace_mode)

        # Final test evaluation uses the validation-selected policy, not
        # merely the parameters from the final optimization step.
        if cfg.workers.final_test_eval and not trace_mode and final_outcomes:
            if self.best_policy_state is None:
                raise RuntimeError(
                    "final test evaluation requested but no validation-selected "
                    "worker checkpoint is available"
                )
            self.policy.load_state_dict(self.best_policy_state)
            test_result = self._evaluate("test", workers.steps)
            logger.info(
                "final test: mean_f1=%.4f mean_em=%.4f",
                test_result.mean_f1 or 0.0, test_result.mean_em or 0.0,
            )

        return {
            "stage_dir": str(self.stage_dir),
            "metrics_path": str(self.metrics_path),
            "updates_completed": len(final_outcomes),
            "best_val_f1": self.best_val_f1,
            "best_step": self.best_step,
            "trace_path": (
                str(self.stage_dir / "trace.json") if trace_mode else None
            ),
            "worked_example_path": (
                str(self.stage_dir / "worked_example.md")
                if trace_mode
                else None
            ),
        }
