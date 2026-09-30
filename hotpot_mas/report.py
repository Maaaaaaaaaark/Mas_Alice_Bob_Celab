"""Report generation: run-level, per-question, and dataset summaries.

All metrics requested by spec sec. 16, computed from the raw JSONL:

- F1 / EM mean and std (population std over the relevant unit);
- generated tokens per agent and total, input tokens total;
- decision steps;
- query counts (Alice / Bob) and message counts;
- Cap Rate, Generation Cap Rate, Natural Termination Rate,
  Parse Error Rate, plus Forced Final Rate and Error Rate;
- termination reason distribution.

The primary token metric is ``total_generated_tokens`` (spec sec. 12).
The three JSON summaries are always written; ``report.md`` is the
human-readable English report.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .logging_io import load_runs


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "n": len(values),
    }


def _rate(runs: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    if not runs:
        return {"rate": 0.0, "count": 0, "n": 0}
    count = sum(1 for run in runs if run.get(key))
    return {"rate": count / len(runs), "count": count, "n": len(runs)}


def _rate_term_reason(
    runs: Sequence[Dict[str, Any]], reason: str
) -> Dict[str, float]:
    if not runs:
        return {"rate": 0.0, "count": 0, "n": 0}
    count = sum(1 for run in runs if run.get("termination_reason") == reason)
    return {"rate": count / len(runs), "count": count, "n": len(runs)}


def _termination_counts(runs: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for run in runs:
        reason = str(run.get("termination_reason"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _mean_std_of_mean(values: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Mean/std over per-question means (dataset-level aggregation)."""
    means = [statistics.fmean(group) for group in values if group]
    return _mean_std(means)


def _run_level_stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean/std of every scalar metric across all runs, plus all rates."""
    first = runs[0] if runs else {}
    architecture = first.get("architecture", "mas")
    clarification_enabled = bool(
        first.get("config", {})
        .get("worker_clarification", {})
        .get("enabled", False)
    )
    shared_question = bool(
        first.get("config", {}).get("share_question_with_workers", False)
    )
    evidence_partition = first.get("config", {}).get(
        "evidence_partition", "supporting_only"
    )
    model_instance_mode = first.get("config", {}).get(
        "model_instance_mode", "shared"
    )
    clarification_requests = sum(
        int(r.get("num_clarification_requests", 0)) for r in runs
    )
    clarification_completed = sum(
        int(r.get("num_clarification_round_trips_completed", 0))
        for r in runs
    )
    stats: Dict[str, Any] = {
        "architecture": architecture,
        "worker_clarification_enabled": clarification_enabled,
        "share_question_with_workers": shared_question,
        "evidence_partition": evidence_partition,
        "partition_seed": first.get("config", {}).get("partition_seed", 0),
        "model_instance_mode": model_instance_mode,
        "num_runs": len(runs),
        "f1": _mean_std([r["f1"] for r in runs]),
        "em": _mean_std([r["em"] for r in runs]),
        "generated_tokens_alice": _mean_std(
            [r["generated_tokens_alice"] for r in runs]
        ),
        "generated_tokens_bob": _mean_std(
            [r["generated_tokens_bob"] for r in runs]
        ),
        "generated_tokens_celab": _mean_std(
            [r["generated_tokens_celab"] for r in runs]
        ),
        "total_generated_tokens": _mean_std(
            [r["total_generated_tokens"] for r in runs]
        ),
        "input_tokens_alice": _mean_std([r["input_tokens_alice"] for r in runs]),
        "input_tokens_bob": _mean_std([r["input_tokens_bob"] for r in runs]),
        "input_tokens_celab": _mean_std([r["input_tokens_celab"] for r in runs]),
        "input_tokens_total": _mean_std([r["input_tokens_total"] for r in runs]),
        "decision_steps": _mean_std([r["decision_steps"] for r in runs]),
        "num_alice_queries": _mean_std([r["num_alice_queries"] for r in runs]),
        "num_bob_queries": _mean_std([r["num_bob_queries"] for r in runs]),
        "num_messages": _mean_std([r["num_messages"] for r in runs]),
        "num_clarification_requests": _mean_std(
            [r.get("num_clarification_requests", 0) for r in runs]
        ),
        "num_clarification_round_trips_completed": _mean_std(
            [
                r.get("num_clarification_round_trips_completed", 0)
                for r in runs
            ]
        ),
        "clarification_round_trip_completion": {
            "rate": (
                clarification_completed / clarification_requests
                if clarification_requests
                else 0.0
            ),
            "count": clarification_completed,
            "n": clarification_requests,
        },
        "rates": {
            "cap_rate": _rate(runs, "cap_reached"),
            "generation_cap_rate": _rate(runs, "generation_cap_reached"),
            "natural_termination_rate": _rate(runs, "natural_termination"),
            # Parse Error Rate counts runs whose termination reason is a
            # main-loop parse error (forced-final parse failures are tracked
            # separately in termination_reason_counts).
            "parse_error_rate": _rate_term_reason(runs, "parse_error"),
            "forced_final_rate": _rate(runs, "forced_final_calls"),
            "unclosed_final_fallback_rate": _rate(
                runs, "final_parse_fallback"
            ),
            "protocol_parse_fallback_rate": _rate(
                runs, "protocol_parse_fallback"
            ),
            "casefold_route_fallback_rate": _rate(
                runs, "casefold_route_fallback"
            ),
            "terminal_final_precedence_fallback_rate": _rate(
                runs, "terminal_final_precedence_fallback"
            ),
            "clarification_trigger_rate": _rate(
                runs, "clarification_triggered"
            ),
            "clarification_protocol_violation_rate": _rate(
                runs, "clarification_protocol_violations"
            ),
            "error_rate": _rate_term_reason(runs, "error"),
        },
        "termination_reason_counts": _termination_counts(runs),
        "token_sum_consistency": {
            "total_generated_tokens_matches_event_sum": bool(
                runs
                and all(
                    r["total_generated_tokens"]
                    == sum(
                        e["generated_tokens"]
                        for e in r["events"]
                        if e["event_type"] == "model_generation"
                    )
                    for r in runs
                )
            )
        },
    }
    return stats


def _per_question_stats(
    runs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """One summary row per question (mean/std over that question's runs)."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["question_id"], []).append(run)

    rows: List[Dict[str, Any]] = []
    for question_id, group in grouped.items():
        first = group[0]
        rows.append(
            {
                "question_id": question_id,
                "question": first["question"],
                "gold_answer": first["gold_answer"],
                "question_type": first["question_type"],
                "num_runs": len(group),
                "f1": _mean_std([r["f1"] for r in group]),
                "em": _mean_std([r["em"] for r in group]),
                "total_generated_tokens": _mean_std(
                    [r["total_generated_tokens"] for r in group]
                ),
                "decision_steps": _mean_std(
                    [r["decision_steps"] for r in group]
                ),
                "num_alice_queries": _mean_std(
                    [r["num_alice_queries"] for r in group]
                ),
                "num_bob_queries": _mean_std(
                    [r["num_bob_queries"] for r in group]
                ),
                "num_clarification_requests": _mean_std(
                    [r.get("num_clarification_requests", 0) for r in group]
                ),
                "clarification_trigger_count": sum(
                    1 for r in group if r.get("clarification_triggered")
                ),
                "parse_error_count": sum(
                    1
                    for r in group
                    if r.get("termination_reason") == "parse_error"
                ),
                "cap_count": sum(1 for r in group if r.get("cap_reached")),
            }
        )
    return rows


def _dataset_level_stats(
    runs: List[Dict[str, Any]],
    per_question: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Dataset-level summary: mean/std over per-question means."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["question_id"], []).append(run)
    groups = list(grouped.values())
    return {
        "num_questions": len(groups),
        "num_runs": len(runs),
        "f1_over_question_means": _mean_std_of_mean(
            [[r["f1"] for r in g] for g in groups]
        ),
        "em_over_question_means": _mean_std_of_mean(
            [[r["em"] for r in g] for g in groups]
        ),
        "total_generated_tokens_over_question_means": _mean_std_of_mean(
            [[r["total_generated_tokens"] for r in g] for g in groups]
        ),
        "decision_steps_over_question_means": _mean_std_of_mean(
            [[r["decision_steps"] for r in g] for g in groups]
        ),
    }


def _markdown_report(
    run_level: Dict[str, Any],
    dataset: Dict[str, Any],
    source_path: Path,
) -> str:
    r = run_level
    rates = r["rates"]
    centralized = r["architecture"] == "centralized_reader"
    if centralized:
        title = "# HotpotQA Centralized Reader Diagnostic — Report"
    elif r["architecture"] == "one_shot_gather":
        title = "# HotpotQA One-Shot Gather MAS Control — Report"
    elif r["architecture"] == "one_shot_direct_answer":
        title = "# HotpotQA One-Shot Direct-Answer MAS Control — Report"
    elif r["worker_clarification_enabled"]:
        title = "# HotpotQA Worker-Clarification MAS Experiment — Report"
    elif r["share_question_with_workers"]:
        title = "# HotpotQA Shared-Question MAS Control — Report"
    else:
        title = "# HotpotQA 3-Agent MAS Base Experiment — Report"
    lines = [
        title,
        "",
        f"source: `{source_path}`",
        f"runs: {r['num_runs']}",
        f"architecture: `{r['architecture']}`",
        f"evidence partition: `{r['evidence_partition']}`",
        f"model instances: `{r['model_instance_mode']}`",
        "",
        "## Answer quality",
        "",
        f"- F1 (mean ± std): {r['f1']['mean']:.4f} ± {r['f1']['std']:.4f}",
        f"- EM (mean ± std): {r['em']['mean']:.4f} ± {r['em']['std']:.4f}",
        "",
        "## Token accounting (primary: total_generated_tokens)",
        "",
        f"- generated tokens, Alice (mean ± std): "
        f"{r['generated_tokens_alice']['mean']:.1f} ± {r['generated_tokens_alice']['std']:.1f}",
        f"- generated tokens, Bob (mean ± std): "
        f"{r['generated_tokens_bob']['mean']:.1f} ± {r['generated_tokens_bob']['std']:.1f}",
        f"- generated tokens, {'Reader' if centralized else 'Celab'} (mean ± std): "
        f"{r['generated_tokens_celab']['mean']:.1f} ± {r['generated_tokens_celab']['std']:.1f}",
        f"- total generated tokens (mean ± std): "
        f"{r['total_generated_tokens']['mean']:.1f} ± {r['total_generated_tokens']['std']:.1f}",
        f"- input tokens total (mean ± std, auxiliary): "
        f"{r['input_tokens_total']['mean']:.1f} ± {r['input_tokens_total']['std']:.1f}",
        "",
        "## Interaction",
        "",
        f"- decision steps (mean ± std): "
        f"{r['decision_steps']['mean']:.2f} ± {r['decision_steps']['std']:.2f}",
        f"- Alice queries (mean ± std): "
        f"{r['num_alice_queries']['mean']:.2f} ± {r['num_alice_queries']['std']:.2f}",
        f"- Bob queries (mean ± std): "
        f"{r['num_bob_queries']['mean']:.2f} ± {r['num_bob_queries']['std']:.2f}",
        f"- messages (mean ± std): "
        f"{r['num_messages']['mean']:.2f} ± {r['num_messages']['std']:.2f}",
        "",
    ]
    if r["worker_clarification_enabled"]:
        completion = r["clarification_round_trip_completion"]
        lines += [
            "## Clarification",
            "",
            f"- clarification requests (mean ± std): "
            f"{r['num_clarification_requests']['mean']:.2f} ± "
            f"{r['num_clarification_requests']['std']:.2f}",
            f"- completed clarification round trips (mean ± std): "
            f"{r['num_clarification_round_trips_completed']['mean']:.2f} ± "
            f"{r['num_clarification_round_trips_completed']['std']:.2f}",
            f"- clarification round-trip completion rate: "
            f"{completion['rate']:.4f} ({completion['count']}/{completion['n']})",
            "",
        ]
    lines += ["## Rates", ""]
    for name, label in [
        ("cap_rate", "Cap Rate (decision cap reached)"),
        ("generation_cap_rate", "Generation Cap Rate"),
        ("natural_termination_rate", "Natural Termination Rate"),
        ("parse_error_rate", "Parse Error Rate"),
        ("forced_final_rate", "Forced Final Rate"),
        ("unclosed_final_fallback_rate", "Unclosed Final Fallback Rate"),
        ("protocol_parse_fallback_rate", "Any Protocol Parse Fallback Rate"),
        ("casefold_route_fallback_rate", "Case-folded Route Fallback Rate"),
        (
            "terminal_final_precedence_fallback_rate",
            "Terminal Final Precedence Fallback Rate",
        ),
    ]:
        entry = rates[name]
        lines.append(
            f"- {label}: {entry['rate']:.4f} ({entry['count']}/{entry['n']})"
        )
    if r["worker_clarification_enabled"]:
        for name, label in [
            ("clarification_trigger_rate", "Clarification Trigger Rate"),
            (
                "clarification_protocol_violation_rate",
                "Clarification Protocol Violation Rate",
            ),
        ]:
            entry = rates[name]
            lines.append(
                f"- {label}: {entry['rate']:.4f} "
                f"({entry['count']}/{entry['n']})"
            )
    entry = rates["error_rate"]
    lines.append(
        f"- Error Rate: {entry['rate']:.4f} ({entry['count']}/{entry['n']})"
    )
    lines += [
        "",
        "## Termination reasons",
        "",
    ]
    for reason, count in sorted(r["termination_reason_counts"].items()):
        lines.append(f"- {reason}: {count}")
    lines += [
        "",
        "## Dataset-level (over per-question means)",
        "",
        f"- F1: {dataset['f1_over_question_means']['mean']:.4f} ± "
        f"{dataset['f1_over_question_means']['std']:.4f}",
        f"- EM: {dataset['em_over_question_means']['mean']:.4f} ± "
        f"{dataset['em_over_question_means']['std']:.4f}",
        f"- total generated tokens: "
        f"{dataset['total_generated_tokens_over_question_means']['mean']:.1f} ± "
        f"{dataset['total_generated_tokens_over_question_means']['std']:.1f}",
        "",
        "## Consistency checks",
        "",
        "- total_generated_tokens equals the sum of model_generation event "
        f"generated_tokens for every run: "
        f"{r['token_sum_consistency']['total_generated_tokens_matches_event_sum']}",
        "",
    ]
    return "\n".join(lines)


def write_report(runs_path: Path, out_dir: Path) -> Dict[str, Path]:
    """Write the three JSON summaries and report.md; return their paths."""
    runs = load_runs(runs_path)
    if not runs:
        raise RuntimeError(f"no runs found in {runs_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_level = _run_level_stats(runs)
    per_question = _per_question_stats(runs)
    dataset = _dataset_level_stats(runs, per_question)

    paths: Dict[str, Path] = {}
    summary_run_level = out_dir / "summary_run_level.json"
    summary_run_level.write_text(
        json.dumps(run_level, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths["summary_run_level"] = summary_run_level

    summary_per_question = out_dir / "summary_per_question.json"
    summary_per_question.write_text(
        json.dumps(per_question, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths["summary_per_question"] = summary_per_question

    summary_dataset = out_dir / "summary_dataset.json"
    summary_dataset.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths["summary_dataset"] = summary_dataset

    report_md = out_dir / "report.md"
    report_md.write_text(
        _markdown_report(run_level, dataset, runs_path), encoding="utf-8"
    )
    paths["report"] = report_md
    return paths
