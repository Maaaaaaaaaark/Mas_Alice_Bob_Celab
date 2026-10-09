"""Deterministic evaluation of workers (Evaluate(theta, C, D)) plus the
report cache.

Workers decode with ``do_sample=False`` during evaluation, then the frozen
synthesizer C scores each (a, b) pair; the mean validation F1 selects the
best worker checkpoint (Algorithm 1 of the TeX). The per-question A/B
report texts are cached on disk keyed by checkpoint step, split, decode
parameters, prompt version and template source, so Stage 2 can score the
original C0 and the fine-tuned Cphi on **identical** cached reports, plus
the empty-report control Evaluate_empty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hotpot_mas.evaluation import exact_match_score
from hotpot_mas.question_selection import SelectedQuestion
from hotpot_mas.seeds import derive_generation_seed

from .config import DecodeConfig, WorkerTrainingConfig
from .policy import Policy
from .prompts_builder import TrainingPrompts
from .synthesizers import Synthesizer


@dataclass
class QuestionEval:
    question_id: str
    f1: float
    em: float
    a_generated_tokens: int
    b_generated_tokens: int
    c_generated_tokens: int
    a_text: str = ""
    b_text: str = ""
    c_raw: str = ""
    c_pred: str = ""
    parsed: Optional[bool] = None  # <FINAL> tag parsed successfully


@dataclass
class SplitEvalResult:
    split: str
    questions: List[QuestionEval] = field(default_factory=list)
    reports_from_cache: bool = False

    @property
    def mean_f1(self) -> Optional[float]:
        if not self.questions:
            return None
        return sum(q.f1 for q in self.questions) / len(self.questions)

    @property
    def mean_em(self) -> Optional[float]:
        if not self.questions:
            return None
        return sum(q.em for q in self.questions) / len(self.questions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "mean_f1": self.mean_f1,
            "mean_em": self.mean_em,
            "num_questions": len(self.questions),
            "reports_from_cache": self.reports_from_cache,
        }


def cache_identity(
    checkpoint_id: str,
    split: str,
    decode: DecodeConfig,
    prompts: TrainingPrompts,
    prompt_dir: Path,
) -> str:
    """Stable cache key: same inputs -> same cached reports."""
    import hashlib

    material = json.dumps(
        {
            "checkpoint_id": checkpoint_id,
            "split": split,
            "decode": decode.to_dict(),
            "prompt_version": prompts.version,
            "prompt_hashes": prompts.hashes,
            "prompt_dir": str(prompt_dir),
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


class EvalReportCache:
    """JSON report cache: question id -> {a_text, b_text, a_ids, b_ids}."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self._data: Dict[str, Dict[str, Any]] = {}
        if cache_path.exists():
            self._data = json.loads(cache_path.read_text(encoding="utf-8"))

    def get(self, question_id: str) -> Optional[Dict[str, Any]]:
        return self._data.get(question_id)

    def put(self, question_id: str, entry: Dict[str, Any]) -> None:
        self._data[question_id] = entry

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _decode_reports(
    question: SelectedQuestion,
    policy: Policy,
    prompts: TrainingPrompts,
    eval_decode: DecodeConfig,
    base_seed: int,
) -> Tuple[str, str, List[int], List[int], int, int]:
    """Deterministically decode one A and one B report for a question."""
    a_messages = prompts.worker_messages(
        "A", question.question, question.evidence_alice
    )
    b_messages = prompts.worker_messages(
        "B", question.question, question.evidence_bob
    )
    a_report = policy.decode_report(
        a_messages,
        derive_generation_seed(base_seed, f"{question.question_id}:A", 0),
        eval_decode,
    )
    b_report = policy.decode_report(
        b_messages,
        derive_generation_seed(base_seed, f"{question.question_id}:B", 0),
        eval_decode,
    )
    return (
        a_report.text,
        b_report.text,
        a_report.token_ids,
        b_report.token_ids,
        a_report.num_tokens,
        b_report.num_tokens,
    )


def generate_report_cache(
    policy: Policy,
    questions: List[SelectedQuestion],
    prompts: TrainingPrompts,
    eval_decode: DecodeConfig,
    base_seed: int,
    cache: EvalReportCache,
) -> EvalReportCache:
    """Decode the deterministic A/B reports for every question and save them.

    Stage 2 (Algorithm 2) builds its own report caches from the frozen best
    worker checkpoint with the same decoding setting used at worker
    evaluation; C0 and Cphi are then scored on these identical cached
    reports. Missing entries are decoded and added, existing entries are
    reused as-is, and the cache is saved after every call.
    """
    for question in questions:
        if cache.get(question.question_id) is not None:
            continue
        (
            a_text,
            b_text,
            a_ids,
            b_ids,
            _a_tokens,
            _b_tokens,
        ) = _decode_reports(
            question,
            policy,
            prompts,
            eval_decode,
            base_seed,
        )
        cache.put(
            question.question_id,
            {
                "a_text": a_text,
                "b_text": b_text,
                "a_ids": a_ids,
                "b_ids": b_ids,
            },
        )
    cache.save()
    return cache


def evaluate_workers(
    policy: Policy,
    synthesizer: Synthesizer,
    questions: List[SelectedQuestion],
    prompts: TrainingPrompts,
    workers_cfg: WorkerTrainingConfig,
    split: str,
    base_seed: int,
    cache: Optional[EvalReportCache] = None,
) -> SplitEvalResult:
    """Evaluate the current workers on one split (cache-aware)."""
    result = SplitEvalResult(split=split)
    for question in questions:
        entry = cache.get(question.question_id) if cache else None
        if entry is not None:
            a_text, b_text = entry["a_text"], entry["b_text"]
            a_ids = entry.get("a_ids", [])
            b_ids = entry.get("b_ids", [])
            a_tokens = len(a_ids)
            b_tokens = len(b_ids)
            result.reports_from_cache = True
        else:
            (
                a_text,
                b_text,
                a_ids,
                b_ids,
                a_tokens,
                b_tokens,
            ) = _decode_reports(
                question,
                policy,
                prompts,
                workers_cfg.eval_decode,
                base_seed,
            )
            if cache is not None:
                cache.put(
                    question.question_id,
                    {
                        "a_text": a_text,
                        "b_text": b_text,
                        "a_ids": a_ids,
                        "b_ids": b_ids,
                    },
                )
        messages = prompts.synthesizer_messages(
            question.question, a_text, b_text
        )
        synth = synthesizer.answer(
            messages,
            question.answer,
            question=question.question,
            a_report=a_text,
            b_report=b_text,
        )
        result.questions.append(
            QuestionEval(
                question_id=question.question_id,
                f1=synth.f1,
                em=exact_match_score(synth.pred_answer, question.answer),
                a_generated_tokens=a_tokens or 0,
                b_generated_tokens=b_tokens or 0,
                c_generated_tokens=synth.generated_tokens,
                a_text=a_text,
                b_text=b_text,
                c_raw=synth.raw_output,
                c_pred=synth.pred_answer,
                parsed=synth.parsed,
            )
        )
    if cache is not None:
        cache.save()
    return result


def evaluate_synthesizer_on_cache(
    synthesizer: Synthesizer,
    questions: List[SelectedQuestion],
    prompts: TrainingPrompts,
    cache: EvalReportCache,
    split: str,
    empty_control: bool = False,
) -> SplitEvalResult:
    """Score a synthesizer on cached reports (or empty-report control).

    With ``empty_control=True`` both worker reports are replaced by the
    empty string (Evaluate_empty of the TeX); the cache then only supplies
    the question keys, not the report texts.
    """
    result = SplitEvalResult(split=split, reports_from_cache=not empty_control)
    for question in questions:
        entry = cache.get(question.question_id)
        if empty_control:
            a_text, b_text = "", ""
        else:
            if entry is None:
                raise RuntimeError(
                    f"no cached reports for {question.question_id} in "
                    f"{cache.cache_path}; run worker evaluation first"
                )
            a_text, b_text = entry["a_text"], entry["b_text"]
        messages = prompts.synthesizer_messages(
            question.question, a_text, b_text
        )
        synth = synthesizer.answer(
            messages,
            question.answer,
            question=question.question,
            a_report=a_text,
            b_report=b_text,
        )
        result.questions.append(
            QuestionEval(
                question_id=question.question_id,
                f1=synth.f1,
                em=exact_match_score(synth.pred_answer, question.answer),
                a_generated_tokens=0,
                b_generated_tokens=0,
                c_generated_tokens=synth.generated_tokens,
                a_text=a_text,
                b_text=b_text,
                c_raw=synth.raw_output,
                c_pred=synth.pred_answer,
                parsed=synth.parsed,
            )
        )
    return result
