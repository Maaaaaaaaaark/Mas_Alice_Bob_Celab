"""Mixture building for Stage 1a worker SFT.

The training mixture targets ~70% answerable / 30% unanswerable AFTER
filtering and dedup. Source weights are relative preferences, not
quotas: samples are never duplicated to meet a target. The achieved
per-source answerability is reported; deviation from the target ratio
is reported, not corrected by throwing examples away.

Trace mode has hard requirements that are checked before training:
an explicitly configured number of positive/negative examples and sources,
present, at least one two-sentence target, single visible document,
and every example human-reviewed. Missing requirements are reported
loudly — the trace is never fabricated by relabeling candidates.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .checks import is_reviewed
from .convert import WorkerExample


def eligible_train(
    examples: List[WorkerExample], require_reviewed: bool
) -> List[WorkerExample]:
    """Training candidates: not excluded, split=train, checks passed."""
    eligible = [
        e
        for e in examples
        if not e.excluded
        and e.split == "train"
        and not e.probe
        and all(info["ok"] for info in e.checks.values())
    ]
    # Answer-string absence is not proof of unanswerability. Therefore
    # automatically swapped MuSiQue sibling paragraphs stay in the audit
    # file but cannot enter SFT until reviewed; official SQuAD 2.0
    # negatives remain eligible.
    eligible = [
        e
        for e in eligible
        if e.answerable or e.source == "squad2" or is_reviewed(e)
    ]
    if require_reviewed:
        eligible = [e for e in eligible if is_reviewed(e)]
    return eligible


def two_sentence_example(example: WorkerExample) -> bool:
    return example.answerable and len(example.evidence) == 2


def source_mixture_report(
    examples: List[WorkerExample],
) -> Dict[str, Any]:
    """Per-source positive/negative counts of a selected set."""
    report: Dict[str, Any] = {"per_source": {}, "total": len(examples)}
    positives = 0
    for example in examples:
        entry = report["per_source"].setdefault(
            example.source,
            {"positive": 0, "negative": 0},
        )
        if example.answerable:
            entry["positive"] += 1
            positives += 1
        else:
            entry["negative"] += 1
    report["answerable"] = positives
    report["unanswerable"] = len(examples) - positives
    report["answerable_ratio"] = (
        positives / len(examples) if examples else 0.0
    )
    return report


def select_mixture(
    examples: List[WorkerExample],
    positives: int,
    negatives: int,
    selection_seed: int,
) -> List[WorkerExample]:
    """Select ``positives`` + ``negatives`` examples deterministically.

    Splits the candidates by label, shuffles each pool with the seed,
    and takes the requested number from each. No duplication. Raises
    when a pool is too small (with the exact shortfall) so a shortfall
    is reported, never papered over.
    """
    pos_pool = [e for e in examples if e.answerable]
    neg_pool = [e for e in examples if not e.answerable]
    if len(pos_pool) < positives:
        raise ValueError(
            f"need {positives} answerable training examples, only "
            f"{len(pos_pool)} eligible (after filtering and dedup)"
        )
    if len(neg_pool) < negatives:
        raise ValueError(
            f"need {negatives} unanswerable training examples, only "
            f"{len(neg_pool)} eligible (after filtering and dedup)"
        )
    rng = random.Random(selection_seed)
    pos_pool = sorted(pos_pool, key=lambda e: e.example_id)
    neg_pool = sorted(neg_pool, key=lambda e: e.example_id)
    rng.shuffle(pos_pool)
    rng.shuffle(neg_pool)
    return pos_pool[:positives] + neg_pool[:negatives]


def _weighted_take(
    pool: List[WorkerExample], count: int, weights: Dict[str, float], seed: int
) -> List[WorkerExample]:
    """Deterministic smooth weighted round-robin without duplication."""
    if len(pool) < count:
        raise ValueError(f"need {count} examples, only {len(pool)} eligible")
    rng = random.Random(seed)
    by_source: Dict[str, List[WorkerExample]] = {}
    for example in sorted(pool, key=lambda e: e.example_id):
        by_source.setdefault(example.source, []).append(example)
    for values in by_source.values():
        rng.shuffle(values)
    scores = {source: 0.0 for source in by_source}
    selected: List[WorkerExample] = []
    while len(selected) < count:
        available = [source for source, values in by_source.items() if values]
        effective = {
            source: max(float(weights.get(source, 1.0)), 0.0)
            for source in available
        }
        if not any(effective.values()):
            effective = {source: 1.0 for source in available}
        total_weight = sum(effective.values())
        for source in available:
            scores[source] += effective[source]
        chosen = min(
            available,
            key=lambda source: (-scores[source], source),
        )
        scores[chosen] -= total_weight
        selected.append(by_source[chosen].pop())
    return selected


def plan_counts(
    examples: List[WorkerExample],
    target_answerable_ratio: float,
    total: int,
) -> Tuple[int, int]:
    """Requested positive/negative counts for a given total."""
    positives = int(round(target_answerable_ratio * total))
    negatives = total - positives
    return positives, negatives


def select_trace(
    examples: List[WorkerExample],
    config: Any,
) -> Tuple[List[WorkerExample], List[str]]:
    """Select and verify the trace set; missing requirements are errors.

    Requirements: exactly ``trace_num_examples`` (default 10) with
    ``trace_positives``/``trace_negatives``, the configured required
    sources present, at least one two-sentence target, single-document
    inputs, every example reviewed.
    """
    eligible = eligible_train(examples, require_reviewed=True)
    if len(eligible) < config.mix.trace_num_examples:
        raise ValueError(
            "trace requires exactly "
            f"{config.mix.trace_num_examples} reviewed training examples; "
            f"only {len(eligible)} eligible. Missing reviewed examples are "
            "reported, never fabricated."
        )
    selected = select_mixture(
        eligible,
        config.mix.trace_positives,
        config.mix.trace_negatives,
        config.mix.selection_seed,
    )
    problems = _trace_requirements_check(
        selected, set(config.mix.trace_required_sources)
    )
    if problems:
        raise ValueError("trace requirements not met: " + "; ".join(problems))
    return selected, problems


def _trace_requirements_check(
    selected: List[WorkerExample], required_sources: set[str]
) -> List[str]:
    problems: List[str] = []
    sources = {e.source for e in selected}
    if not required_sources.issubset(sources):
        problems.append(
            f"required sources {sorted(required_sources)}, "
            f"got {sorted(sources)}"
        )
    if not any(two_sentence_example(e) for e in selected):
        problems.append("at least one two-sentence target required")
    multi_doc = [e.example_id for e in selected if len(e.documents) > 1]
    if multi_doc:
        problems.append(
            f"single-document inputs required, multi-document found: {multi_doc}"
        )
    unreviewed = [e.example_id for e in selected if not is_reviewed(e)]
    if unreviewed:
        problems.append(f"all examples must be reviewed: {unreviewed}")
    return problems


def select_smoke_or_normal(
    examples: List[WorkerExample],
    config: Any,
    require_reviewed: bool,
) -> Tuple[List[WorkerExample], Dict[str, Any]]:
    """Weighted selection for smoke/normal runs.

    Per-source positive/negative counts follow the source weights and
    the target answerable ratio; the achieved mixture is reported.
    """
    eligible = eligible_train(examples, require_reviewed=require_reviewed)
    total = config.mix.total_examples
    if total <= 0:
        raise ValueError("mix.total_examples must be > 0 for this mode")
    positives, negatives = plan_counts(
        eligible, config.mix.target_answerable_ratio, total
    )
    source_weights = {
        name: source.weight
        for name, source in config.data.sources.items()
        if source.enabled
    }
    pos_pool = [e for e in eligible if e.answerable]
    neg_pool = [e for e in eligible if not e.answerable]
    if len(pos_pool) < positives or len(neg_pool) < negatives:
        # Reuse the detailed, label-specific error messages.
        select_mixture(
            eligible, positives, negatives, config.mix.selection_seed
        )
        raise AssertionError("unreachable")
    selected = _weighted_take(
        pos_pool, positives, source_weights, config.mix.selection_seed
    ) + _weighted_take(
        neg_pool, negatives, source_weights, config.mix.selection_seed + 1
    )
    report = source_mixture_report(selected)
    report["requested"] = {"total": total, "positive": positives, "negative": negatives}
    report["source_weights"] = source_weights
    report["note"] = (
        "source weights are relative preferences, not quotas; samples are "
        "never duplicated to meet a target"
    )
    return selected, report
