"""Streaming failure-attribution diagnostics for one-shot MAS runs.

The attribution is deliberately observational.  It follows the normalized
gold-answer string through private evidence, worker reports, and C's final
answer.  This is useful for locating a likely information-loss stage, but it
does not prove a unique causal failure: aliases, paraphrases, and genuinely
multi-hop answers can make lexical tracing incomplete.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .evaluation import normalize_answer


FAILURE_LABELS = {
    "system_or_protocol_failure": "Empty final answer or system/protocol failure",
    "final_answer_overcomplete": "Final answer contains gold but is not exact",
    "synthesis_selection_failure": "Worker report contains gold but C misses it",
    "worker_extraction_failure": "Evidence contains gold but worker reports miss it",
    "answer_not_lexically_explicit_in_evidence": (
        "Gold string is not lexically explicit in private evidence"
    ),
}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def _golds(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _normalized_span_present(text: Any, gold: Any) -> bool:
    """Return whether any normalized gold is a contiguous token subsequence."""
    normalized_text = normalize_answer(str(text or "")).split()
    if not normalized_text:
        return False
    for candidate in _golds(gold):
        normalized_gold = normalize_answer(candidate).split()
        if not normalized_gold:
            continue
        width = len(normalized_gold)
        if any(
            normalized_text[index : index + width] == normalized_gold
            for index in range(len(normalized_text) - width + 1)
        ):
            return True
    return False


def _speaker_outputs(run: Mapping[str, Any]) -> Dict[str, str]:
    outputs: MutableMapping[str, List[str]] = defaultdict(list)
    for event in run.get("events", []):
        if event.get("event_type") != "model_generation":
            continue
        speaker = str(event.get("speaker") or "")
        if speaker:
            outputs[speaker].append(str(event.get("raw_output") or ""))
    return {speaker: "\n".join(parts) for speaker, parts in outputs.items()}


def _presence_bucket(alice: bool, bob: bool) -> str:
    if alice and bob:
        return "both"
    if alice:
        return "alice_only"
    if bob:
        return "bob_only"
    return "neither"


def _primary_failure_category(
    run: Mapping[str, Any],
    *,
    final_has_gold: bool,
    report_has_gold: bool,
    evidence_has_gold: bool,
) -> str:
    final_answer = str(run.get("final_answer") or "").strip()
    termination = run.get("termination_reason")
    if (
        run.get("error")
        or not final_answer
        or termination not in {"direct_answer", "natural_final"}
    ):
        return "system_or_protocol_failure"
    if final_has_gold:
        return "final_answer_overcomplete"
    if report_has_gold:
        return "synthesis_selection_failure"
    if evidence_has_gold:
        return "worker_extraction_failure"
    return "answer_not_lexically_explicit_in_evidence"


def _safe_rate(count: int, denominator: int) -> float:
    return count / denominator if denominator else 0.0


def analyze_failure_attribution(
    runs_path: Path,
    *,
    samples_per_category: int = 5,
) -> Dict[str, Any]:
    """Stream one JSONL file and return a serializable attribution summary."""
    runs_path = Path(runs_path)
    total_runs = 0
    malformed_lines = 0
    seen_ids = set()
    question_stats: Dict[str, Dict[str, Any]] = {}
    outcome_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    evidence_location_counts: Counter[str] = Counter()
    report_location_counts: Counter[str] = Counter()
    report_bucket_exact: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    secondary_counts: Counter[str] = Counter()
    category_f1: MutableMapping[str, List[float]] = defaultdict(list)
    category_tokens: MutableMapping[str, List[float]] = defaultdict(list)
    samples: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    sampled_question_ids: MutableMapping[str, set[str]] = defaultdict(set)
    expected_seed_set: set[int] | None = None

    with runs_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue

            run_id = str(run.get("run_id") or "")
            if not run_id:
                raise ValueError(f"missing run_id at {runs_path}:{line_number}")
            if run_id in seen_ids:
                raise ValueError(f"duplicate run_id {run_id!r} in {runs_path}")
            seen_ids.add(run_id)

            if expected_seed_set is None:
                configured = run.get("config", {}).get("run_seeds")
                if isinstance(configured, Sequence) and not isinstance(
                    configured, (str, bytes)
                ):
                    expected_seed_set = {int(seed) for seed in configured}

            total_runs += 1
            gold = run.get("gold_answer", "")
            outputs = _speaker_outputs(run)
            alice_report = outputs.get("alice", "")
            bob_report = outputs.get("bob", "")
            final_answer = run.get("final_answer", "")
            alice_evidence = run.get(
                "private_evidence_alice", run.get("alice_private_context", "")
            )
            bob_evidence = run.get(
                "private_evidence_bob", run.get("bob_private_context", "")
            )

            alice_evidence_has = _normalized_span_present(alice_evidence, gold)
            bob_evidence_has = _normalized_span_present(bob_evidence, gold)
            alice_report_has = _normalized_span_present(alice_report, gold)
            bob_report_has = _normalized_span_present(bob_report, gold)
            final_has = _normalized_span_present(final_answer, gold)
            evidence_has = alice_evidence_has or bob_evidence_has
            report_has = alice_report_has or bob_report_has

            evidence_bucket = _presence_bucket(
                alice_evidence_has, bob_evidence_has
            )
            report_bucket = _presence_bucket(alice_report_has, bob_report_has)
            evidence_location_counts[evidence_bucket] += 1
            report_location_counts[report_bucket] += 1

            em = float(run.get("em", 0.0))
            f1 = float(run.get("f1", 0.0))
            exact = em == 1.0
            if exact:
                outcome = "exact_success"
                report_bucket_exact[report_bucket] += 1
            elif f1 > 0.0:
                outcome = "non_exact_partial_overlap"
            else:
                outcome = "zero_overlap_failure"
            outcome_counts[outcome] += 1

            if evidence_has:
                stage_counts["gold_span_in_any_private_evidence"] += 1
            if report_has:
                stage_counts["gold_span_in_any_worker_report"] += 1
            if final_has:
                stage_counts["gold_span_in_final_answer"] += 1
            if exact:
                stage_counts["official_exact_match"] += 1

            if run.get("generation_cap_reached"):
                secondary_counts["generation_cap_reached"] += 1
            if run.get("parse_error"):
                secondary_counts["parse_error"] += 1
            if run.get("error"):
                secondary_counts["runtime_error"] += 1

            question_id = str(run.get("question_id") or "")
            qstat = question_stats.setdefault(
                question_id,
                {"runs": 0, "exact": 0, "seeds": set(), "failures": Counter()},
            )
            qstat["runs"] += 1
            qstat["exact"] += int(exact)
            qstat["seeds"].add(int(run.get("run_seed", -1)))

            if exact:
                continue

            category = _primary_failure_category(
                run,
                final_has_gold=final_has,
                report_has_gold=report_has,
                evidence_has_gold=evidence_has,
            )
            failure_counts[category] += 1
            qstat["failures"][category] += 1
            category_f1[category].append(f1)
            category_tokens[category].append(
                float(run.get("total_generated_tokens", 0.0))
            )
            if (
                len(samples[category]) < samples_per_category
                and question_id not in sampled_question_ids[category]
            ):
                samples[category].append(
                    {
                        "run_id": run_id,
                        "question_id": question_id,
                        "run_seed": int(run.get("run_seed", -1)),
                        "question": run.get("question"),
                        "gold_answer": gold,
                        "final_answer": final_answer,
                        "f1": f1,
                        "evidence_location": evidence_bucket,
                        "report_location": report_bucket,
                    }
                )
                sampled_question_ids[category].add(question_id)

    failures = total_runs - outcome_counts["exact_success"]
    categories = {}
    for category in FAILURE_LABELS:
        count = failure_counts[category]
        categories[category] = {
            "label": FAILURE_LABELS[category],
            "count": count,
            "rate_of_failures": _safe_rate(count, failures),
            "rate_of_all_runs": _safe_rate(count, total_runs),
            "mean_f1": _mean(category_f1[category]),
            "mean_total_generated_tokens": _mean(category_tokens[category]),
            "sample_runs": samples[category],
        }

    expected_seed_set = expected_seed_set or set()
    completed_questions = 0
    never_exact_completed = 0
    always_exact_completed = 0
    at_least_one_exact_completed = 0
    for qstat in question_stats.values():
        complete = bool(expected_seed_set) and qstat["seeds"] == expected_seed_set
        if not complete:
            continue
        completed_questions += 1
        if qstat["exact"] == 0:
            never_exact_completed += 1
        else:
            at_least_one_exact_completed += 1
        if qstat["exact"] == len(expected_seed_set):
            always_exact_completed += 1

    report_bucket_stats = {}
    for bucket in ("alice_only", "bob_only", "both", "neither"):
        count = report_location_counts[bucket]
        report_bucket_stats[bucket] = {
            "count": count,
            "rate": _safe_rate(count, total_runs),
            "exact_matches": report_bucket_exact[bucket],
            "exact_match_rate": _safe_rate(report_bucket_exact[bucket], count),
        }

    return {
        "source": str(runs_path.resolve()),
        "method": "normalized contiguous gold-span tracing",
        "method_limitations": (
            "Heuristic observational attribution, not causal proof. Aliases, "
            "paraphrases, one-token answers, and multi-hop inference can cause "
            "lexical presence to under- or over-estimate information transfer."
        ),
        "total_runs": total_runs,
        "unique_questions": len(question_stats),
        "malformed_lines_ignored": malformed_lines,
        "outcomes": {
            name: {
                "count": outcome_counts[name],
                "rate": _safe_rate(outcome_counts[name], total_runs),
            }
            for name in (
                "exact_success",
                "non_exact_partial_overlap",
                "zero_overlap_failure",
            )
        },
        "failed_runs": failures,
        "primary_failure_categories": categories,
        "lexical_stage_funnel": {
            name: {
                "count": stage_counts[name],
                "rate": _safe_rate(stage_counts[name], total_runs),
            }
            for name in (
                "gold_span_in_any_private_evidence",
                "gold_span_in_any_worker_report",
                "gold_span_in_final_answer",
                "official_exact_match",
            )
        },
        "gold_span_location_in_private_evidence": {
            bucket: {
                "count": evidence_location_counts[bucket],
                "rate": _safe_rate(evidence_location_counts[bucket], total_runs),
            }
            for bucket in ("alice_only", "bob_only", "both", "neither")
        },
        "gold_span_location_in_worker_reports": report_bucket_stats,
        "secondary_flags": {
            name: {
                "count": secondary_counts[name],
                "rate": _safe_rate(secondary_counts[name], total_runs),
            }
            for name in (
                "generation_cap_reached",
                "parse_error",
                "runtime_error",
            )
        },
        "question_level": {
            "questions_seen": len(question_stats),
            "expected_seeds": sorted(expected_seed_set),
            "questions_with_all_expected_seeds": completed_questions,
            "complete_questions_with_no_exact_seed": never_exact_completed,
            "complete_questions_with_at_least_one_exact_seed": (
                at_least_one_exact_completed
            ),
            "complete_questions_exact_on_every_seed": always_exact_completed,
        },
    }


def _markdown_report(summary: Mapping[str, Any]) -> str:
    total = int(summary["total_runs"])
    failures = int(summary["failed_runs"])
    lines = [
        "# HotpotQA One-Shot MAS Failure Attribution — Report",
        "",
        f"source: `{summary['source']}`",
        f"runs analyzed: {total}",
        f"questions observed: {summary['unique_questions']}",
        "",
        "## Method and scope",
        "",
        "This report traces the normalized gold-answer string through Alice's "
        "and Bob's private evidence, their generated reports, and C's final "
        "answer. Each non-EM run is assigned to the earliest observable lexical "
        "information-loss stage.",
        "",
        f"**Important limitation:** {summary['method_limitations']}",
        "",
        "## Outcomes",
        "",
        "| Outcome | Count | Rate |",
        "|---|---:|---:|",
    ]
    outcome_labels = {
        "exact_success": "Official exact match",
        "non_exact_partial_overlap": "Non-exact with partial token overlap",
        "zero_overlap_failure": "Zero token-overlap failure",
    }
    for key, label in outcome_labels.items():
        item = summary["outcomes"][key]
        lines.append(f"| {label} | {item['count']} | {item['rate']:.4f} |")

    lines += [
        "",
        "## Primary attribution among non-EM runs",
        "",
        f"Non-EM runs: {failures}",
        "",
        "| Attribution | Count | Share of failures | Share of all runs | Mean F1 | Mean generated tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in FAILURE_LABELS:
        item = summary["primary_failure_categories"][key]
        lines.append(
            f"| {item['label']} | {item['count']} | "
            f"{item['rate_of_failures']:.4f} | {item['rate_of_all_runs']:.4f} | "
            f"{item['mean_f1']:.4f} | "
            f"{item['mean_total_generated_tokens']:.1f} |"
        )

    lines += [
        "",
        "## Lexical information funnel",
        "",
        "| Observable stage | Count | Rate over all runs |",
        "|---|---:|---:|",
    ]
    funnel_labels = {
        "gold_span_in_any_private_evidence": "Gold span in A or B private evidence",
        "gold_span_in_any_worker_report": "Gold span in A or B report",
        "gold_span_in_final_answer": "Gold span in C final answer",
        "official_exact_match": "Official exact match",
    }
    for key, label in funnel_labels.items():
        item = summary["lexical_stage_funnel"][key]
        lines.append(f"| {label} | {item['count']} | {item['rate']:.4f} |")

    lines += [
        "",
        "## Where is the gold span in private evidence?",
        "",
        "| Evidence location | Runs | Rate |",
        "|---|---:|---:|",
    ]
    bucket_labels = {
        "alice_only": "Alice only",
        "bob_only": "Bob only",
        "both": "Both",
        "neither": "Neither",
    }
    for key, label in bucket_labels.items():
        item = summary["gold_span_location_in_private_evidence"][key]
        lines.append(f"| {label} | {item['count']} | {item['rate']:.4f} |")

    lines += [
        "",
        "## Which worker report contains the gold span?",
        "",
        "| Report location | Runs | Rate | EM within bucket |",
        "|---|---:|---:|---:|",
    ]
    for key, label in bucket_labels.items():
        item = summary["gold_span_location_in_worker_reports"][key]
        lines.append(
            f"| {label} | {item['count']} | {item['rate']:.4f} | "
            f"{item['exact_match_rate']:.4f} |"
        )

    question = summary["question_level"]
    lines += [
        "",
        "## Question-level stability",
        "",
        f"- questions with all expected seeds: "
        f"{question['questions_with_all_expected_seeds']}",
        f"- complete questions with no exact seed: "
        f"{question['complete_questions_with_no_exact_seed']}",
        f"- complete questions with at least one exact seed: "
        f"{question['complete_questions_with_at_least_one_exact_seed']}",
        f"- complete questions exact on every seed: "
        f"{question['complete_questions_exact_on_every_seed']}",
        "",
        "## Secondary execution flags",
        "",
    ]
    for key, item in summary["secondary_flags"].items():
        lines.append(f"- {key}: {item['count']} ({item['rate']:.4f})")

    lines += ["", "## Example run IDs by attribution", ""]
    for key in FAILURE_LABELS:
        item = summary["primary_failure_categories"][key]
        lines.append(f"### {item['label']}")
        lines.append("")
        if not item["sample_runs"]:
            lines.append("- none")
        else:
            for sample in item["sample_runs"]:
                lines.append(
                    f"- `{sample['run_id']}` — F1 {sample['f1']:.4f}; "
                    f"gold: `{sample['gold_answer']}`; final: "
                    f"`{sample['final_answer']}`"
                )
        lines.append("")
    return "\n".join(lines)


def write_failure_attribution_report(
    runs_path: Path,
    output_dir: Path,
    *,
    samples_per_category: int = 5,
) -> Dict[str, Path]:
    """Analyze MAS failures and write JSON plus an English Markdown report."""
    summary = analyze_failure_attribution(
        runs_path, samples_per_category=samples_per_category
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "failure_attribution_summary.json"
    report_path = output_dir / "failure_attribution_report.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_path.write_text(_markdown_report(summary), encoding="utf-8")
    return {
        "failure_attribution_summary": summary_path,
        "failure_attribution_report": report_path,
    }
