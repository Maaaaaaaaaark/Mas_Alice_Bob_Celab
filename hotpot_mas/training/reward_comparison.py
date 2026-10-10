"""Compare EM, F1, and gold-answer log-likelihood rewards on 50 questions."""

from __future__ import annotations

import gc
import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from hotpot_mas.evaluation import exact_match_score
from hotpot_mas.question_selection import SelectedQuestion, load_manifest
from hotpot_mas.seeds import derive_generation_seed, seed_all

from .config import TrainingConfig
from .cross_pair import compute_marginals
from .data import manifest_path_for_split
from .inference_diagnostic import normalized_span_present
from .policy import HFPolicy
from .prompts_builder import TrainingPrompts
from .synthesizers import HFSynthesizer

logger = logging.getLogger(__name__)
REWARDS = ("em", "f1", "gold_mean_log_likelihood")


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        output[str(row["question_id"])] = row
    return output


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def summarize_reward_rows(
    rows: Iterable[Mapping[str, Any]], delta: float
) -> Dict[str, Any]:
    rows = list(rows)
    summary: Dict[str, Any] = {
        "num_questions": len(rows),
        "signal_delta": delta,
        "rewards": {},
    }
    for reward in REWARDS:
        signal_any = 0
        operational_signal = 0
        report_rewards: List[float] = []
        contains: List[float] = []
        all_values: List[float] = []
        side_stds: List[float] = []
        for row in rows:
            matrix = row["reward_matrices"][reward]
            q_a, q_b, std_a, std_b = compute_marginals(matrix)
            side_stds.extend([std_a, std_b])
            signal_any += int(std_a > 1e-8 or std_b > 1e-8)
            operational_signal += int(std_a > delta or std_b > delta)
            all_values.extend(
                float(value)
                for matrix_row in matrix
                for value in matrix_row
            )
            a_reports = row["reports"]["a"]
            b_reports = row["reports"]["b"]
            for report, value in zip(a_reports, q_a):
                contains.append(float(bool(report["contains_normalized_gold"])))
                report_rewards.append(float(value))
            for report, value in zip(b_reports, q_b):
                contains.append(float(bool(report["contains_normalized_gold"])))
                report_rewards.append(float(value))
        positive = [r for r, flag in zip(report_rewards, contains) if flag]
        negative = [r for r, flag in zip(report_rewards, contains) if not flag]
        summary["rewards"][reward] = {
            "mean_pair_reward": statistics.fmean(all_values) if all_values else None,
            "questions_with_any_variance": signal_any,
            "any_variance_rate": signal_any / len(rows) if rows else None,
            "questions_with_signal_at_delta": operational_signal,
            "signal_rate_at_delta": operational_signal / len(rows) if rows else None,
            "median_side_std": (
                statistics.median(side_stds) if side_stds else None
            ),
            "report_contains_gold_correlation": _pearson(contains, report_rewards),
            "mean_marginal_if_report_contains_gold": (
                statistics.fmean(positive) if positive else None
            ),
            "mean_marginal_if_report_misses_gold": (
                statistics.fmean(negative) if negative else None
            ),
            "num_reports_containing_gold": int(sum(contains)),
            "num_reports": len(contains),
        }
    def correlation_or_floor(name: str) -> float:
        value = summary["rewards"][name][
            "report_contains_gold_correlation"
        ]
        return float(value) if value is not None else -1.0

    ranked = sorted(
        REWARDS,
        key=lambda name: (
            float(summary["rewards"][name]["any_variance_rate"] or 0.0),
            correlation_or_floor(name),
        ),
        reverse=True,
    )
    summary["ranking_rule"] = (
        "descending questions-with-any-variance rate, then descending "
        "correlation between report gold containment and marginal reward"
    )
    summary["reward_ranking"] = ranked
    summary["provisional_recommendation"] = ranked[0] if ranked else None
    return summary


def render_reward_summary(summary: Mapping[str, Any]) -> str:
    """Human-readable comparison; JSON remains the source of truth."""
    lines = [
        "# Reward comparison on fixed train questions",
        "",
        f"Questions: {summary['num_questions']}",
        f"Signal threshold delta: {summary['signal_delta']}",
        "",
        "| Reward | Any variance | Signal at delta | Containment correlation |",
        "|---|---:|---:|---:|",
    ]
    for name in REWARDS:
        row = summary["rewards"][name]
        correlation = row["report_contains_gold_correlation"]
        correlation_text = (
            "—" if correlation is None else f"{float(correlation):.4f}"
        )
        lines.append(
            f"| {name} | {row['questions_with_any_variance']} "
            f"({float(row['any_variance_rate'] or 0.0):.4f}) | "
            f"{row['questions_with_signal_at_delta']} "
            f"({float(row['signal_rate_at_delta'] or 0.0):.4f}) | "
            f"{correlation_text} |"
        )
    lines.extend(
        [
            "",
            f"Ranking: {summary['reward_ranking']}",
            f"Provisional recommendation: "
            f"{summary['provisional_recommendation']}",
            "",
            "The recommendation follows the recorded ranking rule; inspect "
            "all columns before selecting the learner reward.",
            "",
        ]
    )
    return "\n".join(lines)


class RewardComparison:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.output_dir = cfg.stage_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.output_dir / "sampled_reports.json"
        self.results_path = self.output_dir / "per_question.jsonl"
        self.prompts = TrainingPrompts(cfg.prompt_dir)

    def _generate_reports(self, questions: List[SelectedQuestion]) -> Dict[str, Any]:
        cache = (
            json.loads(self.cache_path.read_text(encoding="utf-8"))
            if self.cache_path.exists()
            else {}
        )
        missing = [q for q in questions if q.question_id not in cache]
        if not missing:
            return cache
        seed_all(self.cfg.seed)
        policy = HFPolicy(self.cfg.model)
        g = self.cfg.workers.G
        for q_index, question in enumerate(missing, 1):
            entry: Dict[str, List[Dict[str, Any]]] = {"a": [], "b": []}
            for side, evidence, key in (
                ("A", question.evidence_alice, "a"),
                ("B", question.evidence_bob, "b"),
            ):
                messages = self.prompts.worker_messages(
                    side,
                    question.question,
                    evidence,
                )
                attempt = 0
                while len(entry[key]) < g and attempt < g * 20:
                    seed = derive_generation_seed(
                        self.cfg.seed,
                        f"reward-compare:{question.question_id}:{side}",
                        attempt,
                    )
                    report = policy.sample_report(
                        messages,
                        seed,
                        self.cfg.workers.rollout,
                    )
                    attempt += 1
                    if not report.token_ids or report.finish_reason != "eos":
                        continue
                    entry[key].append(
                        {
                            "text": report.text,
                            "token_ids": report.token_ids,
                            "generated_tokens": report.num_tokens,
                            "finish_reason": report.finish_reason,
                            "generation_seed": seed,
                            "contains_normalized_gold": normalized_span_present(
                                report.text, question.answer
                            ),
                        }
                    )
                if len(entry[key]) != g:
                    raise RuntimeError(
                        f"could not obtain {g} completed {side} reports for "
                        f"{question.question_id} in {g * 20} attempts"
                    )
            cache[question.question_id] = entry
            _save_json(self.cache_path, cache)
            if q_index % 5 == 0 or q_index == len(missing):
                logger.info(
                    "reward comparison reports: %d/%d",
                    q_index,
                    len(missing),
                )
        del policy
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return cache

    def run(self) -> Dict[str, Any]:
        questions = load_manifest(manifest_path_for_split(self.cfg, "train"))
        reports = self._generate_reports(questions)
        existing = _load_jsonl(self.results_path)
        seed_all(self.cfg.seed)
        synthesizer = HFSynthesizer(
            self.cfg.model, self.cfg.workers.synthesizer_decode
        )
        g = self.cfg.workers.G
        for q_index, question in enumerate(questions, 1):
            if question.question_id in existing:
                continue
            entry = reports[question.question_id]
            matrices = {
                reward: [[0.0 for _ in range(g)] for _ in range(g)]
                for reward in REWARDS
            }
            pair_outputs: List[Dict[str, Any]] = []
            for i in range(g):
                for j in range(g):
                    a_text = entry["a"][i]["text"]
                    b_text = entry["b"][j]["text"]
                    messages = self.prompts.synthesizer_messages(
                        question.question, a_text, b_text
                    )
                    answer = synthesizer.answer(
                        messages,
                        question.answer,
                        question=question.question,
                        a_report=a_text,
                        b_report=b_text,
                    )
                    likelihood = synthesizer.gold_answer_log_likelihood(
                        messages, question.answer
                    )
                    em = exact_match_score(answer.pred_answer, question.answer)
                    matrices["em"][i][j] = em
                    matrices["f1"][i][j] = answer.f1
                    matrices["gold_mean_log_likelihood"][i][j] = likelihood[
                        "mean_log_likelihood"
                    ]
                    pair_outputs.append(
                        {
                            "i": i,
                            "j": j,
                            "prediction": answer.pred_answer,
                            "raw_output": answer.raw_output,
                            "parsed": answer.parsed,
                            "f1": answer.f1,
                            "em": em,
                            "gold_likelihood": likelihood,
                        }
                    )
            row = {
                "question_id": question.question_id,
                "question": question.question,
                "gold_answer": question.answer,
                "reports": entry,
                "reward_matrices": matrices,
                "pair_outputs": pair_outputs,
            }
            _append_jsonl(self.results_path, row)
            existing[question.question_id] = row
            if q_index % 5 == 0 or q_index == len(questions):
                logger.info(
                    "reward comparison questions: %d/%d",
                    q_index,
                    len(questions),
                )

        summary = summarize_reward_rows(existing.values(), self.cfg.workers.delta)
        summary["config"] = self.cfg.to_dict()
        summary["complete"] = len(existing) == len(questions)
        summary_path = self.output_dir / "summary.json"
        _save_json(summary_path, summary)
        report_path = self.output_dir / "report.md"
        report_path.write_text(
            render_reward_summary(summary), encoding="utf-8"
        )
        return {
            "output_dir": str(self.output_dir),
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            "results_path": str(self.results_path),
            "complete": summary["complete"],
        }
