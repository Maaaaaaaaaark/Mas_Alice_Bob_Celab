"""Paired comparisons across MAS controls and the centralized reader."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .logging_io import load_runs


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def _key(run: Dict[str, Any]) -> Tuple[str, int]:
    return str(run["question_id"]), int(run["run_seed"])


def _index(runs: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    indexed = {_key(run): run for run in runs}
    if len(indexed) != len(runs):
        raise ValueError("duplicate question_id/run_seed pair in diagnostic input")
    return indexed


def _validate_compatibility(groups: Dict[str, List[Dict[str, Any]]]) -> None:
    for name, runs in groups.items():
        if not runs:
            raise ValueError(f"{name} runs file is empty")
    reference = groups["baseline"][0]
    for name, runs in groups.items():
        first = runs[0]
        for field in ("model_name", "model_revision", "tokenizer_revision"):
            if first.get(field) != reference.get(field):
                raise ValueError(f"{name} differs from baseline on {field}")


def _validate_matched_content(
    indexes: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]],
    common: List[Tuple[str, int]],
) -> None:
    for key in common:
        baseline = indexes["baseline"][key]
        for name in indexes:
            if name == "baseline":
                continue
            condition = indexes[name][key]
            for field in ("question", "gold_answer"):
                if (
                    field in baseline
                    and field in condition
                    and baseline[field] != condition[field]
                ):
                    raise ValueError(
                        f"{name} differs from baseline on {field} for {key}"
                    )


def _bootstrap_question_delta(
    baseline: Dict[Tuple[str, int], Dict[str, Any]],
    condition: Dict[Tuple[str, int], Dict[str, Any]],
    metric: str,
    samples: int = 5000,
) -> Dict[str, Any]:
    common = sorted(set(baseline) & set(condition))
    if not common:
        raise ValueError("diagnostic conditions have no matching runs")
    by_question: Dict[str, List[float]] = {}
    for key in common:
        by_question.setdefault(key[0], []).append(
            float(condition[key][metric]) - float(baseline[key][metric])
        )
    question_deltas = [
        statistics.fmean(values) for values in by_question.values()
    ]
    observed = statistics.fmean(question_deltas)
    rng = random.Random(0)
    boot = []
    for _ in range(samples):
        resample = [
            question_deltas[rng.randrange(len(question_deltas))]
            for _ in question_deltas
        ]
        boot.append(statistics.fmean(resample))
    boot.sort()
    low = boot[int(0.025 * (samples - 1))]
    high = boot[int(0.975 * (samples - 1))]
    return {
        "matched_runs": len(common),
        "matched_questions": len(by_question),
        "mean_delta": observed,
        "bootstrap_95_ci": [low, high],
    }


def _condition_summary(runs: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "runs": len(runs),
        "questions": len({str(run["question_id"]) for run in runs}),
        "f1": _mean(float(run["f1"]) for run in runs),
        "em": _mean(float(run["em"]) for run in runs),
        "generated_tokens": _mean(
            float(run["total_generated_tokens"]) for run in runs
        ),
        "input_tokens": _mean(float(run["input_tokens_total"]) for run in runs),
        "parse_error_rate": _mean(
            run.get("termination_reason") == "parse_error" for run in runs
        ),
    }


def write_diagnostic_report(
    baseline_path: Path,
    clarification_path: Path,
    centralized_path: Path,
    output_dir: Path,
    shared_question_path: Path | None = None,
    one_shot_path: Path | None = None,
) -> Dict[str, Path]:
    """Write an English paired diagnostic report and its JSON payload."""
    groups = {
        "baseline": load_runs(baseline_path),
        "clarification": load_runs(clarification_path),
    }
    if shared_question_path is not None:
        groups["shared_question"] = load_runs(shared_question_path)
    if one_shot_path is not None:
        groups["one_shot_gather"] = load_runs(one_shot_path)
    groups["centralized_reader"] = load_runs(centralized_path)
    _validate_compatibility(groups)
    indexes = {name: _index(runs) for name, runs in groups.items()}
    common = sorted(set.intersection(*(set(index) for index in indexes.values())))
    if not common:
        raise ValueError("the diagnostic conditions have no matching runs")
    _validate_matched_content(indexes, common)
    matched_indexes = {
        name: {key: index[key] for key in common}
        for name, index in indexes.items()
    }
    summaries = {
        name: _condition_summary([matched_indexes[name][key] for key in common])
        for name in groups
    }
    comparisons: Dict[str, Dict[str, Any]] = {}
    conditions = [name for name in groups if name != "baseline"]
    for condition in conditions:
        comparisons[condition] = {
            metric: _bootstrap_question_delta(
                matched_indexes["baseline"], matched_indexes[condition], metric
            )
            for metric in ("f1", "em", "total_generated_tokens")
        }

    staged_comparisons: Dict[str, Dict[str, Any]] = {}
    if "shared_question" in matched_indexes:
        staged_comparisons["question_visibility"] = {
            metric: _bootstrap_question_delta(
                matched_indexes["baseline"],
                matched_indexes["shared_question"],
                metric,
            )
            for metric in ("f1", "em", "total_generated_tokens")
        }
    if {
        "shared_question",
        "one_shot_gather",
    }.issubset(matched_indexes):
        staged_comparisons["fixed_schedule"] = {
            metric: _bootstrap_question_delta(
                matched_indexes["shared_question"],
                matched_indexes["one_shot_gather"],
                metric,
            )
            for metric in ("f1", "em", "total_generated_tokens")
        }
    if "one_shot_gather" in matched_indexes:
        staged_comparisons["centralization"] = {
            metric: _bootstrap_question_delta(
                matched_indexes["one_shot_gather"],
                matched_indexes["centralized_reader"],
                metric,
            )
            for metric in ("f1", "em", "total_generated_tokens")
        }

    central_delta = comparisons["centralized_reader"]["f1"]
    central_f1 = summaries["centralized_reader"]["f1"]
    low, high = central_delta["bootstrap_95_ci"]
    if low > 0 and central_f1 >= 0.5:
        diagnosis = "communication coordination is the dominant observed bottleneck"
        rationale = (
            "Direct access to both evidence documents produces a reliable F1 gain "
            "and the direct reader reaches at least 0.50 mean F1."
        )
    elif low > 0:
        diagnosis = "both communication coordination and direct reading are bottlenecks"
        rationale = (
            "Direct access produces a reliable F1 gain, while the direct reader "
            "still remains below 0.50 mean F1."
        )
    elif high <= 0:
        diagnosis = "direct reading or answer extraction is the dominant observed bottleneck"
        rationale = (
            "Removing communication does not improve F1 within the paired "
            "bootstrap interval."
        )
    else:
        diagnosis = "the pilot is inconclusive about the dominant bottleneck"
        rationale = (
            "The paired 95% bootstrap interval for the centralized-reader F1 "
            "difference includes zero."
        )

    payload = {
        "source_run_counts": {
            name: len(runs) for name, runs in groups.items()
        },
        "matched_run_count": len(common),
        "matched_question_count": len({key[0] for key in common}),
        "conditions": summaries,
        "paired_comparisons_against_baseline": comparisons,
        "staged_paired_comparisons": staged_comparisons,
        "diagnosis": diagnosis,
        "diagnosis_rationale": rationale,
        "diagnosis_rule_note": (
            "The 0.50 direct-reader F1 boundary is a declared heuristic; the "
            "paired estimates and confidence intervals are the primary evidence."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "diagnostic_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# HotpotQA Reading-vs-Communication Diagnostic — Report",
        "",
        f"Matched comparison set: {len(common)} runs over "
        f"{len({key[0] for key in common})} questions.",
        "All condition means below use this same matched set.",
        "",
        "## Conditions",
        "",
        "| Condition | Runs | Questions | F1 | EM | Generated tokens | Input tokens | Parse error rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "baseline": "MAS baseline",
        "clarification": "Worker clarification",
        "centralized_reader": "Centralized reader",
        "shared_question": "Shared-question MAS",
        "one_shot_gather": "One-shot gather MAS",
    }
    for name in groups:
        item = summaries[name]
        lines.append(
            f"| {labels[name]} | {int(item['runs'])} | "
            f"{int(item['questions'])} | {item['f1']:.4f} | "
            f"{item['em']:.4f} | {item['generated_tokens']:.1f} | "
            f"{item['input_tokens']:.1f} | {item['parse_error_rate']:.4f} |"
        )
    lines += ["", "## Paired differences from the MAS baseline", ""]
    for condition in conditions:
        lines.append(f"### {labels[condition]}")
        lines.append("")
        for metric in ("f1", "em", "total_generated_tokens"):
            item = comparisons[condition][metric]
            ci = item["bootstrap_95_ci"]
            lines.append(
                f"- {metric}: mean delta {item['mean_delta']:+.4f}; "
                f"question-bootstrap 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}] "
                f"over {item['matched_runs']} matched runs and "
                f"{item['matched_questions']} questions"
            )
        lines.append("")
    if staged_comparisons:
        lines += ["## Staged mechanism comparisons", ""]
        stage_labels = {
            "question_visibility": (
                "Shared-question MAS minus MAS baseline "
                "(effect of worker question visibility)"
            ),
            "fixed_schedule": (
                "One-shot gather minus shared-question MAS "
                "(effect of replacing free routing with a fixed schedule)"
            ),
            "centralization": (
                "Centralized reader minus one-shot gather MAS "
                "(remaining evidence-transfer/centralization gap)"
            ),
        }
        for stage, metrics in staged_comparisons.items():
            lines.append(f"### {stage_labels[stage]}")
            lines.append("")
            for metric in ("f1", "em", "total_generated_tokens"):
                item = metrics[metric]
                ci = item["bootstrap_95_ci"]
                lines.append(
                    f"- {metric}: mean delta {item['mean_delta']:+.4f}; "
                    f"question-bootstrap 95% CI "
                    f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                )
            lines.append("")
    lines += [
        "## Diagnostic interpretation",
        "",
        f"**Result: {diagnosis}.**",
        "",
        rationale,
        "",
        "The 0.50 direct-reader F1 boundary is a declared heuristic. The paired "
        "differences and their question-level bootstrap intervals should be used "
        "as the primary evidence. This diagnostic does not prove a unique causal "
        "decomposition: one-shot gather changes the interaction schedule and "
        "dialogue opportunity, while the centralized reader also changes evidence "
        "access and prompting.",
        "",
    ]
    report_path = output_dir / "diagnostic_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"diagnostic_summary": summary_path, "diagnostic_report": report_path}
