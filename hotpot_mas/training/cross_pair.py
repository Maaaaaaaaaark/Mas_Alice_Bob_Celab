"""Cross-paired reward aggregation (pure functions, Algorithm 1 of the TeX).

Given the G x G reward matrix R where ``R[i][j] = F1(C(q, a_i, b_j), y*)``,
this module computes the marginal rewards Q_A/Q_B, the per-side standard
deviations, the signal flags, and the normalized advantages. It contains no
model or dataset code so the arithmetic can be unit-tested offline.

All standard deviations here are **population** standard deviations
(divide by G, not G - 1), matching the TeX notation ``std(Q)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _pop_std(values: Sequence[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def compute_marginals(
    r_matrix: Sequence[Sequence[float]],
) -> Tuple[List[float], List[float], float, float]:
    """Return ``(q_a, q_b, std_a, std_b)`` for a square reward matrix."""
    g = len(r_matrix)
    if g < 1:
        raise ValueError("reward matrix must be non-empty")
    for row in r_matrix:
        if len(row) != g:
            raise ValueError("reward matrix must be square (G x G)")
    q_a = [_mean(row) for row in r_matrix]
    q_b = [
        _mean([r_matrix[i][j] for i in range(g)]) for j in range(g)
    ]
    return q_a, q_b, _pop_std(q_a), _pop_std(q_b)


def signal_flags(std_a: float, std_b: float, delta: float) -> Tuple[bool, bool]:
    """Signal test: a side signals iff ``std(Q_side) > delta``."""
    return std_a > delta, std_b > delta


def normalize_advantages(
    values: Sequence[float], eps_n: float
) -> List[float]:
    """Normalize one side's marginal rewards to advantages.

    ``Adv = (Q - mean(Q)) / (std(Q) + eps_n)``. Only called on signal
    sides, whose population std is strictly positive; the guard below
    fails loudly instead of dividing by ``eps_n`` alone.
    """
    if len(values) == 0:
        raise ValueError("cannot normalize empty marginal rewards")
    mean = _mean(values)
    std = _pop_std(values)
    if std <= 0.0:
        raise ValueError(
            "cannot normalize a zero-variance side; a signal side must "
            "have std > delta >= 0"
        )
    return [(v - mean) / (std + eps_n) for v in values]


@dataclass
class SignalResult:
    """Aggregated cross-paired reward information for one question."""

    r_matrix: List[List[float]]
    q_a: List[float] = field(default_factory=list)
    q_b: List[float] = field(default_factory=list)
    std_a: float = 0.0
    std_b: float = 0.0
    signal_a: bool = False
    signal_b: bool = False
    kept_sides: List[str] = field(default_factory=list)  # subset of A,B
    advantages_a: Optional[List[float]] = None
    advantages_b: Optional[List[float]] = None

    @property
    def has_signal(self) -> bool:
        return bool(self.kept_sides)

    @property
    def kept_report_count(self) -> int:
        """|S_q|: G (single side) or 2G (both sides)."""
        return len(self.r_matrix) * len(self.kept_sides)

    def advantage_for(self, side: str, index: int) -> float:
        """Normalized advantage of one report on a kept side."""
        if side == "A":
            if not self.signal_a or self.advantages_a is None:
                raise ValueError("side A did not signal")
            return self.advantages_a[index]
        if side == "B":
            if not self.signal_b or self.advantages_b is None:
                raise ValueError("side B did not signal")
            return self.advantages_b[index]
        raise ValueError(f"unknown side {side!r}")


def build_signal_result(
    r_matrix: Sequence[Sequence[float]], delta: float, eps_n: float
) -> SignalResult:
    """Aggregate a reward matrix into marginal rewards and advantages.

    Sides with ``std > delta`` signal; the kept set S_q holds the reports
    of every signalling side. A question with no signalling side keeps
    nothing (the caller discards or re-samples it).
    """
    rows = [list(row) for row in r_matrix]
    q_a, q_b, std_a, std_b = compute_marginals(rows)
    signal_a, signal_b = signal_flags(std_a, std_b, delta)
    kept: List[str] = []
    if signal_a:
        kept.append("A")
    if signal_b:
        kept.append("B")
    result = SignalResult(
        r_matrix=rows,
        q_a=q_a,
        q_b=q_b,
        std_a=std_a,
        std_b=std_b,
        signal_a=signal_a,
        signal_b=signal_b,
        kept_sides=kept,
    )
    if signal_a:
        result.advantages_a = normalize_advantages(q_a, eps_n)
    if signal_b:
        result.advantages_b = normalize_advantages(q_b, eps_n)
    return result
