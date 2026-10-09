"""Cross-paired reward aggregation: marginals, signals, advantages."""

from __future__ import annotations

import pytest

from hotpot_mas.training.cross_pair import (
    build_signal_result,
    compute_marginals,
    normalize_advantages,
    signal_flags,
)


class TestMarginals:
    def test_row_and_column_means(self):
        r = [[1.0, 0.0], [0.0, 0.5]]
        q_a, q_b, std_a, std_b = compute_marginals(r)
        assert q_a == pytest.approx([0.5, 0.25])
        assert q_b == pytest.approx([0.5, 0.25])
        assert std_a == pytest.approx(0.125)
        assert std_b == pytest.approx(0.125)

    def test_flat_matrix_has_zero_std(self):
        r = [[0.5, 0.5], [0.5, 0.5]]
        _, _, std_a, std_b = compute_marginals(r)
        assert std_a == pytest.approx(0.0)
        assert std_b == pytest.approx(0.0)

    def test_population_std_used(self):
        # std over [1, 0] as population: sqrt(0.25) = 0.5, not /1
        _, _, std_a, _ = compute_marginals([[1.0, 1.0], [0.0, 0.0]])
        assert std_a == pytest.approx(0.5)

    def test_non_square_rejected(self):
        with pytest.raises(ValueError, match="square"):
            compute_marginals([[1.0, 0.0]])

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_marginals([])


class TestSignals:
    def test_single_side_signal_keeps_only_that_side(self):
        r = [[1.0, 1.0], [0.0, 0.0]]  # q_a varies, q_b flat
        result = build_signal_result(r, delta=0.05, eps_n=1e-6)
        assert result.signal_a is True
        assert result.signal_b is False
        assert result.kept_sides == ["A"]
        assert result.kept_report_count == 2

    def test_both_sides_signal_keeps_both(self):
        r = [[1.0, 0.0], [0.0, 0.0]]  # both vary
        result = build_signal_result(r, delta=0.05, eps_n=1e-6)
        assert result.signal_a is True
        assert result.signal_b is True
        assert result.kept_sides == ["A", "B"]
        assert result.kept_report_count == 4  # 2G

    def test_no_signal_keeps_nothing(self):
        r = [[0.2, 0.2], [0.2, 0.2]]
        result = build_signal_result(r, delta=0.05, eps_n=1e-6)
        assert result.has_signal is False
        assert result.kept_sides == []
        assert result.kept_report_count == 0
        assert result.advantages_a is None
        assert result.advantages_b is None

    def test_signal_flags_strict_inequality(self):
        assert signal_flags(0.05, 0.04, 0.05) == (False, False)
        assert signal_flags(0.0500001, 0.0, 0.05) == (True, False)


class TestAdvantages:
    def test_normalization_matches_tex(self):
        # Q = [1, 0]: mean 0.5, std 0.5 -> adv = (Q - 0.5) / (0.5 + eps)
        adv = normalize_advantages([1.0, 0.0], eps_n=1e-6)
        assert adv == pytest.approx([0.999998, -0.999998], abs=1e-4)

    def test_advantages_sum_to_zero(self):
        adv = normalize_advantages([0.9, 0.4, 0.2], eps_n=1e-6)
        assert sum(adv) == pytest.approx(0.0, abs=1e-9)

    def test_zero_variance_rejected(self):
        with pytest.raises(ValueError, match="zero-variance"):
            normalize_advantages([0.5, 0.5], eps_n=1e-6)

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            normalize_advantages([], eps_n=1e-6)

    def test_advantage_for_unsignalled_side_raises(self):
        r = [[1.0, 1.0], [0.0, 0.0]]
        result = build_signal_result(r, delta=0.05, eps_n=1e-6)
        with pytest.raises(ValueError, match="did not signal"):
            result.advantage_for("B", 0)

    def test_reward_matrix_copied_not_aliased(self):
        r = [[1.0, 0.0], [0.0, 0.0]]
        result = build_signal_result(r, delta=0.05, eps_n=1e-6)
        r[0][0] = 0.0
        assert result.r_matrix[0][0] == 1.0
