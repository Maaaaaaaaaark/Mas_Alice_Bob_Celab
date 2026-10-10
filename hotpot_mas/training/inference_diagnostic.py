"""Four-condition, inference-only diagnostic on a fixed validation split.

This module deliberately does not load a trained checkpoint or run backward.
It measures how much performance is lost at the worker-report interface by
holding the base model, question set, answer protocol, and greedy decoding
fixed across four evidence conditions.
"""

from __future__ import annotations

import gc
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from hotpot_mas.evaluation import exact_match_score, normalize_answer
from hotpot_mas.prompts import PromptSet
from hotpot_mas.question_selection import SelectedQuestion, load_manifest
from hotpot_mas.seeds import derive_generation_seed, seed_all

from .config import TrainingConfig
from .data import manifest_path_for_split
from .eval import EvalReportCache
from .policy import HFPolicy
from .prompts_builder import TrainingPrompts
from .synthesizers import HFSynthesizer

logger = logging.getLogger(__name__)

CONDITIONS = (
    "empty_reports",
    "untrained_worker_reports",
    "gold_supporting_documents",
    "single_agent_all_documents",
)

CONDITION_LABELS = {
    "empty_reports": "两份空报告",
    "untrained_worker_reports": "未训练 worker 的报告",
    "gold_supporting_documents": "两篇 supporting 文档",
    "single_agent_all_documents": "全部 10 篇文档，单 agent 作答",
}


def normalized_span_present(text: str, gold: str) -> bool:
    """Whether normalized gold tokens occur contiguously in ``text``."""
    haystack = normalize_answer(str(text or "")).split()
    needle = normalize_answer(str(gold or "")).split()
    if not haystack or not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(
        haystack[index : index + width] == needle
        for index in range(len(haystack) - width + 1)
    )


def literal_string_present(text: str, gold: str) -> bool:
    """Case-insensitive gold-string containment after whitespace cleanup."""
    haystack = " ".join(str(text or "").split()).casefold()
    needle = " ".join(str(gold or "").split()).casefold()
    return bool(needle) and needle in haystack


def prompt_echo_present(text: str) -> bool:
    """Conservative detector for recognizable instruction repetition."""
    normalized = normalize_answer(str(text or ""))
    phrases = (
        "you are evidence worker",
        "original question",
        "private passages",
        "some passages may be unrelated",
        "identify entities and relations",
        "preserve exact names dates and numbers",
        "do not speculate",
        "repeat instructions",
        "write one short factual report",
        "include only information relevant",
    )
    return any(normalize_answer(phrase) in normalized for phrase in phrases)


def _render_document(document: Mapping[str, Any]) -> str:
    return f"Title: {document['title']}\n{document['paragraph']}"


def gold_supporting_reports(question: SelectedQuestion) -> Tuple[str, str]:
    """Return the two full gold documents, ordered like worker A then B.

    Gold labels are used only to construct this oracle diagnostic condition;
    labels themselves are never rendered into the model input.
    """
    supporting = {
        str(document["title"]): document
        for document in question.document_pool
        if bool(document.get("is_supporting"))
    }
    if len(question.supporting_titles) != 2:
        raise RuntimeError(
            f"question {question.question_id} does not have two supporting titles"
        )
    missing = [title for title in question.supporting_titles if title not in supporting]
    if missing:
        raise RuntimeError(
            f"question {question.question_id} is missing supporting "
            f"documents: {missing}"
        )
    return (
        _render_document(supporting[question.supporting_titles[0]]),
        _render_document(supporting[question.supporting_titles[1]]),
    )


def direct_reader_messages(
    prompt_set: PromptSet, question: str, evidence_all: str
) -> List[Dict[str, str]]:
    """Dedicated single-reader prompt over all ten candidate documents."""
    return [
        {
            "role": "system",
            "content": prompt_set.render("direct_reader_system"),
        },
        {
            "role": "user",
            "content": (
                "Question:\nIs the stated claim true?\n\n"
                "Candidate documents:\nThe document explicitly confirms the claim."
            ),
        },
        {"role": "assistant", "content": "Answer: yes"},
        {
            "role": "user",
            "content": (
                "Question:\nIn what year did the event occur?\n\n"
                "Candidate documents:\nThe event occurred in 1969."
            ),
        },
        {"role": "assistant", "content": "Answer: 1969"},
        {
            "role": "user",
            "content": (
                "Question:\nIs the object made of wood?\n\n"
                "Candidate documents:\nIt is made of metal, not wood."
            ),
        },
        {"role": "assistant", "content": "Answer: no"},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Candidate documents:\n{evidence_all}"
            ),
        },
    ]


def _load_jsonl(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not path.exists():
        return records
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = (str(record["question_id"]), str(record["condition"]))
        if key in records:
            raise RuntimeError(f"duplicate result {key} at {path}:{line_number}")
        records[key] = record
    return records


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def summarize_records(
    records: Iterable[Mapping[str, Any]], expected_questions: int
) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    for record in records:
        condition = str(record["condition"])
        if condition in grouped:
            grouped[condition].append(record)

    conditions: Dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = grouped[condition]
        conditions[condition] = {
            "label": CONDITION_LABELS[condition],
            "num_questions": len(rows),
            "complete": len(rows) == expected_questions,
            "mean_f1": statistics.fmean(float(row["f1"]) for row in rows)
            if rows else None,
            "mean_em": statistics.fmean(float(row["em"]) for row in rows)
            if rows else None,
            "parse_success_rate": statistics.fmean(
                float(bool(row["parsed"])) for row in rows
            ) if rows else None,
            "mean_c_generated_tokens": statistics.fmean(
                float(row["c_generated_tokens"]) for row in rows
            ) if rows else None,
            "mean_normalized_answer_words": statistics.fmean(
                float(row["normalized_answer_words"]) for row in rows
            ) if rows else None,
            "mean_gold_answer_words": statistics.fmean(
                len(normalize_answer(str(row["gold_answer"])).split())
                for row in rows
            ) if rows else None,
            "mean_absolute_answer_word_error": statistics.fmean(
                abs(
                    float(row["normalized_answer_words"])
                    - len(normalize_answer(str(row["gold_answer"])).split())
                )
                for row in rows
            ) if rows else None,
        }

    worker_rows = grouped["untrained_worker_reports"]
    containment = {
        "num_questions": len(worker_rows),
        "alice_gold_string_rate": statistics.fmean(
            float(bool(row["alice_contains_gold_string"]))
            for row in worker_rows
        ) if worker_rows else None,
        "bob_gold_string_rate": statistics.fmean(
            float(bool(row["bob_contains_gold_string"]))
            for row in worker_rows
        ) if worker_rows else None,
        "either_worker_gold_string_rate": statistics.fmean(
            float(
                bool(row["alice_contains_gold_string"])
                or bool(row["bob_contains_gold_string"])
            )
            for row in worker_rows
        ) if worker_rows else None,
        "alice_normalized_gold_span_rate": statistics.fmean(
            float(bool(row["alice_contains_normalized_gold"]))
            for row in worker_rows
        ) if worker_rows else None,
        "bob_normalized_gold_span_rate": statistics.fmean(
            float(bool(row["bob_contains_normalized_gold"]))
            for row in worker_rows
        ) if worker_rows else None,
        "alice_prompt_echo_rate": statistics.fmean(
            float(bool(row.get("alice_prompt_echo", False)))
            for row in worker_rows
        ) if worker_rows else None,
        "bob_prompt_echo_rate": statistics.fmean(
            float(bool(row.get("bob_prompt_echo", False)))
            for row in worker_rows
        ) if worker_rows else None,
    }
    f1_order = sorted(
        CONDITIONS,
        key=lambda name: float(conditions[name]["mean_f1"] or 0.0),
        reverse=True,
    )
    core_order_matches = (
        float(conditions["gold_supporting_documents"]["mean_f1"] or 0.0)
        > float(conditions["untrained_worker_reports"]["mean_f1"] or 0.0)
        > float(conditions["empty_reports"]["mean_f1"] or 0.0)
    )
    answer_length_close = all(
        (
            float(conditions[name]["mean_absolute_answer_word_error"] or 0.0)
            <= 2.0
            and float(conditions[name]["mean_normalized_answer_words"] or 0.0)
            <= float(conditions[name]["mean_gold_answer_words"] or 0.0) + 2.0
        )
        for name in CONDITIONS
    )
    return {
        "expected_questions": expected_questions,
        "complete": all(
            conditions[condition]["complete"] for condition in CONDITIONS
        ),
        "conditions": conditions,
        "f1_order_best_to_worst": f1_order,
        "expected_core_f1_order_best_to_worst": [
            "gold_supporting_documents",
            "untrained_worker_reports",
            "empty_reports",
        ],
        "single_agent_order_note": (
            "single_agent_all_documents is a separate reference and is "
            "reported, not forced above the oracle two-document condition"
        ),
        "acceptance": {
            "all_parse_rates_at_least_0_95": all(
                float(conditions[name]["parse_success_rate"] or 0.0) >= 0.95
                for name in CONDITIONS
            ),
            "all_em_strictly_positive": all(
                float(conditions[name]["mean_em"] or 0.0) > 0.0
                for name in CONDITIONS
            ),
            "answer_length_close_to_gold": answer_length_close,
            "core_f1_order_matches_expected_table": core_order_matches,
        },
        "untrained_worker_gold_answer_containment": containment,
    }


def render_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Validation inference diagnostic",
        "",
        f"Questions requested: {summary['expected_questions']}",
        f"Complete: {summary['complete']}",
        "",
        "答案长度主列为 C 实际生成的 token 数；"
        "最后一列额外给出解析后答案的规范化词数。",
        "",
        "| C 的输入 | 题数 | F1 | EM | 解析成功率 | "
        "C 生成 token | 解析答案词数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = summary["conditions"][condition]
        def fmt(value: Any) -> str:
            return "—" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {row['label']} | {row['num_questions']} | "
            f"{fmt(row['mean_f1'])} | {fmt(row['mean_em'])} | "
            f"{fmt(row['parse_success_rate'])} | "
            f"{fmt(row['mean_c_generated_tokens'])} | "
            f"{fmt(row['mean_normalized_answer_words'])} |"
        )
    containment = summary["untrained_worker_gold_answer_containment"]
    lines.extend(
        [
            "",
            "## Gold-answer string in untrained-worker reports",
            "",
            "The primary rate is case-insensitive literal-string containment "
            "after whitespace cleanup. Normalized HotpotQA span rates are "
            "also recorded in summary.json.",
            "",
            f"- Alice: {containment['alice_gold_string_rate']}",
            f"- Bob: {containment['bob_gold_string_rate']}",
            f"- Either worker: {containment['either_worker_gold_string_rate']}",
            f"- Alice prompt echo: {containment['alice_prompt_echo_rate']}",
            f"- Bob prompt echo: {containment['bob_prompt_echo_rate']}",
            "",
            "## Acceptance checks",
            "",
            f"- F1 order (best to worst): "
            f"{summary['f1_order_best_to_worst']}",
            f"- Parse rate >= 95%: "
            f"{summary['acceptance']['all_parse_rates_at_least_0_95']}",
            f"- EM > 0 in every condition: "
            f"{summary['acceptance']['all_em_strictly_positive']}",
            f"- Answer length close to gold: "
            f"{summary['acceptance']['answer_length_close_to_gold']}",
            f"- Core order oracle > worker > empty: "
            f"{summary['acceptance']['core_f1_order_matches_expected_table']}",
            "",
        ]
    )
    return "\n".join(lines)


class InferenceDiagnostic:
    """Resumable runner for the four inference-only conditions."""

    def __init__(
        self,
        cfg: TrainingConfig,
        report_cache_path: Path | None = None,
        baseline_summary_path: Path | None = None,
    ):
        self.cfg = cfg
        self.output_dir = cfg.stage_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / "per_question.jsonl"
        self.report_cache = EvalReportCache(
            report_cache_path
            if report_cache_path is not None
            else self.output_dir / "untrained_worker_reports.json"
        )
        self.prompts = TrainingPrompts(cfg.prompt_dir)
        self.reader_prompts = PromptSet(
            cfg.prompt_dir, ("direct_reader_system",)
        )
        self.baseline_summary_path = baseline_summary_path

    def _ensure_untrained_reports(
        self, questions: List[SelectedQuestion]
    ) -> None:
        missing = [
            question for question in questions
            if self.report_cache.get(question.question_id) is None
        ]
        if not missing:
            logger.info("reusing cached untrained-worker reports (%d)", len(questions))
            return
        logger.info("generating %d missing untrained-worker report pairs", len(missing))
        seed_all(self.cfg.seed)
        policy = HFPolicy(self.cfg.model)
        for index, question in enumerate(missing, 1):
            reports: Dict[str, Any] = {}
            for side, evidence in (
                ("A", question.evidence_alice),
                ("B", question.evidence_bob),
            ):
                messages = self.prompts.worker_messages(
                    side, question.question, evidence
                )
                report = policy.decode_report(
                    messages,
                    derive_generation_seed(
                        self.cfg.seed, f"{question.question_id}:{side}", 0
                    ),
                    self.cfg.workers.eval_decode,
                )
                reports[side] = {
                    "text": report.text,
                    "token_ids": report.token_ids,
                    "generated_tokens": report.num_tokens,
                    "finish_reason": report.finish_reason,
                }
            self.report_cache.put(
                question.question_id,
                {"a": reports["A"], "b": reports["B"]},
            )
            self.report_cache.save()
            if index % 10 == 0 or index == len(missing):
                logger.info("untrained reports: %d/%d", index, len(missing))
        del policy
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _condition_input(
        self, condition: str, question: SelectedQuestion
    ) -> Tuple[List[Dict[str, str]], str, str]:
        if condition == "empty_reports":
            a_text, b_text = "", ""
            messages = self.prompts.synthesizer_messages(
                question.question, a_text, b_text
            )
        elif condition == "untrained_worker_reports":
            cached = self.report_cache.get(question.question_id)
            if cached is None:
                raise RuntimeError(f"missing reports for {question.question_id}")
            a_text = str(cached["a"]["text"])
            b_text = str(cached["b"]["text"])
            messages = self.prompts.synthesizer_messages(
                question.question, a_text, b_text
            )
        elif condition == "gold_supporting_documents":
            a_text, b_text = gold_supporting_reports(question)
            messages = self.prompts.synthesizer_messages(
                question.question, a_text, b_text
            )
        elif condition == "single_agent_all_documents":
            a_text, b_text = question.evidence_all, ""
            messages = direct_reader_messages(
                self.reader_prompts, question.question, question.evidence_all
            )
        else:
            raise ValueError(f"unknown diagnostic condition: {condition}")
        return messages, a_text, b_text

    def run(self) -> Dict[str, Any]:
        questions = load_manifest(manifest_path_for_split(self.cfg, "val"))
        expected = self.cfg.data.val.num_questions
        if len(questions) != expected:
            raise RuntimeError(
                f"validation manifest has {len(questions)}, expected {expected}"
            )
        self._ensure_untrained_reports(questions)

        existing = _load_jsonl(self.results_path)
        seed_all(self.cfg.seed)
        synthesizer = HFSynthesizer(
            self.cfg.model, self.cfg.workers.synthesizer_decode
        )
        total = len(questions) * len(CONDITIONS)
        completed = len(existing)
        logger.info("diagnostic answers already complete: %d/%d", completed, total)
        for question_index, question in enumerate(questions, 1):
            for condition in CONDITIONS:
                key = (question.question_id, condition)
                if key in existing:
                    continue
                messages, a_text, b_text = self._condition_input(
                    condition, question
                )
                result = synthesizer.answer(
                    messages,
                    question.answer,
                    question=question.question,
                    a_report=a_text,
                    b_report=b_text,
                )
                cached = self.report_cache.get(question.question_id) or {}
                record: Dict[str, Any] = {
                    "question_id": question.question_id,
                    "question": question.question,
                    "gold_answer": question.answer,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "prediction": result.pred_answer,
                    "raw_output": result.raw_output,
                    "f1": result.f1,
                    "em": exact_match_score(result.pred_answer, question.answer),
                    "parsed": result.parsed,
                    "c_input_tokens": result.input_tokens,
                    "c_generated_tokens": result.generated_tokens,
                    "normalized_answer_words": len(
                        normalize_answer(result.pred_answer).split()
                    ),
                    "finish_reason": result.finish_reason,
                }
                if condition == "untrained_worker_reports":
                    record.update(
                        {
                            "alice_report": a_text,
                            "bob_report": b_text,
                            "alice_report_tokens": cached["a"]["generated_tokens"],
                            "bob_report_tokens": cached["b"]["generated_tokens"],
                            "alice_contains_gold_string": literal_string_present(
                                a_text, question.answer
                            ),
                            "bob_contains_gold_string": literal_string_present(
                                b_text, question.answer
                            ),
                            "alice_contains_normalized_gold": normalized_span_present(
                                a_text, question.answer
                            ),
                            "bob_contains_normalized_gold": normalized_span_present(
                                b_text, question.answer
                            ),
                            "alice_prompt_echo": prompt_echo_present(a_text),
                            "bob_prompt_echo": prompt_echo_present(b_text),
                        }
                    )
                _append_jsonl(self.results_path, record)
                existing[key] = record
                completed += 1
            if question_index % 10 == 0 or question_index == len(questions):
                logger.info(
                    "diagnostic questions: %d/%d; condition results: %d/%d",
                    question_index,
                    len(questions),
                    completed,
                    total,
                )

        summary = summarize_records(existing.values(), expected)
        if self.baseline_summary_path is not None:
            baseline = json.loads(
                self.baseline_summary_path.read_text(encoding="utf-8")
            )
            baseline_order = baseline.get("f1_order_best_to_worst")
            if baseline_order is None:
                baseline_conditions = baseline["conditions"]
                baseline_order = sorted(
                    CONDITIONS,
                    key=lambda name: float(
                        baseline_conditions[name]["mean_f1"] or 0.0
                    ),
                    reverse=True,
                )
            summary["baseline_f1_order_best_to_worst"] = baseline_order
            summary["f1_order_matches_baseline"] = (
                summary["f1_order_best_to_worst"] == baseline_order
            )
        summary.update(
            {
                "config": self.cfg.to_dict(),
                "prompt_version": self.prompts.version,
                "prompt_hashes": self.prompts.hashes,
                "direct_reader_prompt_hashes": self.reader_prompts.hashes,
                "results_path": str(self.results_path),
                "worker_report_cache": str(self.report_cache.cache_path),
                "baseline_summary_path": (
                    str(self.baseline_summary_path)
                    if self.baseline_summary_path is not None
                    else None
                ),
            }
        )
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        report_path = self.output_dir / "report.md"
        report_path.write_text(render_summary(summary), encoding="utf-8")
        return {
            "output_dir": str(self.output_dir),
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            "results_path": str(self.results_path),
            "complete": summary["complete"],
        }
