"""Single-question rollout for the cross-paired GRPO trainer.

For one question: sample G reports from Alice and G from Bob under the
current (old) policy, run the frozen synthesizer C on **all G x G pairs**
to build the reward matrix, compute the marginal rewards and signal flags,
and keep only the reports of signalling sides (S_q). Questions without any
signalling side are discarded by the trainer and re-sampled, bounded by
``max_sampling_attempts``.

The reward for pair (i, j) is the official HotpotQA token F1 between C's
answer on (a_i, b_j) and the gold answer. The gold answer never enters any
worker or synthesizer input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hotpot_mas.evaluation import exact_match_score
from hotpot_mas.question_selection import SelectedQuestion
from hotpot_mas.seeds import derive_generation_seed

from .config import DecodeConfig, WorkerTrainingConfig
from .cross_pair import SignalResult, build_signal_result
from .policy import Policy, Report
from .prompts_builder import TrainingPrompts
from .synthesizers import SynthResult, Synthesizer


@dataclass
class RolloutReport:
    """One sampled report plus everything needed to re-score it later."""

    side: str  # "A" | "B"
    index: int  # 0 .. G-1 within the side
    generation_seed: int
    messages: List[Dict[str, str]]
    prompt_ids: List[int]
    report: Report
    advantage: Optional[float] = None  # set for kept (signal-side) reports


@dataclass
class PairResult:
    """One C call: question + A report i + B report j -> scored answer."""

    i: int
    j: int
    messages: List[Dict[str, str]]
    synth: SynthResult
    reward: float


@dataclass
class QuestionRollout:
    """Full cross-paired rollout for one question."""

    question: SelectedQuestion
    a_reports: List[RolloutReport]
    b_reports: List[RolloutReport]
    pairs: List[List[PairResult]]  # [i][j], G x G
    signal: SignalResult
    kept_reports: List[RolloutReport]  # S_q (signal-side reports only)
    excluded_empty_reports: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_signal(self) -> bool:
        return bool(self.kept_reports)

    def reward_matrix(self) -> List[List[float]]:
        return [[pair.reward for pair in row] for row in self.pairs]


def pair_reward(
    kind: str,
    synthesizer: Synthesizer,
    synth: SynthResult,
    messages: List[Dict[str, str]],
    gold: str,
) -> float:
    """Return the configured scalar reward for one A/B report pair."""
    if kind == "f1":
        return float(synth.f1)
    if kind == "em":
        return float(exact_match_score(synth.pred_answer, gold))
    if kind == "gold_mean_log_likelihood":
        likelihood_fn = getattr(
            synthesizer, "gold_answer_log_likelihood", None
        )
        if likelihood_fn is None:
            raise TypeError(
                "gold_mean_log_likelihood reward requires a synthesizer "
                "with gold_answer_log_likelihood()"
            )
        value = likelihood_fn(messages, gold)
        return float(value["mean_log_likelihood"])
    raise ValueError(f"unsupported reward kind: {kind!r}")


def rollout_question(
    question: SelectedQuestion,
    policy: Policy,
    synthesizer: Synthesizer,
    prompts: TrainingPrompts,
    workers: WorkerTrainingConfig,
    base_seed: int,
    rollout_index: int = 0,
) -> QuestionRollout:
    """Sample and score one question end to end."""
    worker_messages = {
        "A": prompts.worker_messages(
            "A", question.question, question.evidence_alice
        ),
        "B": prompts.worker_messages(
            "B", question.question, question.evidence_bob
        ),
    }
    rollout_reports: Dict[str, List[RolloutReport]] = {}
    for side in ("A", "B"):
        side_reports: List[RolloutReport] = []
        for index in range(workers.G):
            generation_seed = derive_generation_seed(
                base_seed,
                f"{question.question_id}:{side}:rollout-{rollout_index}",
                index,
            )
            report = policy.sample_report(
                worker_messages[side], generation_seed, workers.rollout
            )
            side_reports.append(
                RolloutReport(
                    side=side,
                    index=index,
                    generation_seed=generation_seed,
                    messages=worker_messages[side],
                    prompt_ids=policy.tokenize(worker_messages[side]),
                    report=report,
                )
            )
        rollout_reports[side] = side_reports

    pairs: List[List[PairResult]] = []
    for i in range(workers.G):
        row: List[PairResult] = []
        for j in range(workers.G):
            messages = prompts.synthesizer_messages(
                question.question,
                rollout_reports["A"][i].report.text,
                rollout_reports["B"][j].report.text,
            )
            synth = synthesizer.answer(
                messages,
                question.answer,
                question=question.question,
                a_report=rollout_reports["A"][i].report.text,
                b_report=rollout_reports["B"][j].report.text,
            )
            reward = pair_reward(
                workers.reward_kind,
                synthesizer,
                synth,
                messages,
                question.answer,
            )
            row.append(
                PairResult(
                    i=i,
                    j=j,
                    messages=messages,
                    synth=synth,
                    reward=reward,
                )
            )
        pairs.append(row)

    r_matrix = [[pair.reward for pair in row] for row in pairs]
    signal = build_signal_result(
        r_matrix, workers.delta, workers.eps_n
    )

    kept: List[RolloutReport] = []
    excluded: List[Dict[str, Any]] = []
    for side in signal.kept_sides:
        for rr in rollout_reports[side]:
            if not rr.report.token_ids:
                # Zero-length reports are scored by C (empty string is a
                # legitimate evaluation input) but cannot participate in
                # the token-level loss; exclude them from S_q and record
                # the exclusion so the trace shows it happened.
                excluded.append(
                    {
                        "side": side,
                        "index": rr.index,
                        "reason": "empty_report",
                    }
                )
                continue
            if rr.report.finish_reason != "eos":
                # A max-token cutoff is not a completed report in the task
                # environment. It may be scored for diagnostics, but it must
                # never contribute a policy gradient.
                excluded.append(
                    {
                        "side": side,
                        "index": rr.index,
                        "reason": "truncated_at_max_new_tokens",
                    }
                )
                continue
            rr.advantage = signal.advantage_for(side, rr.index)
            kept.append(rr)

    return QuestionRollout(
        question=question,
        a_reports=rollout_reports["A"],
        b_reports=rollout_reports["B"],
        pairs=pairs,
        signal=signal,
        kept_reports=kept,
        excluded_empty_reports=excluded,
    )
