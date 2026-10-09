"""GRPO loss math: mask handling, ratio/clip, weighting, diagnostics."""

from __future__ import annotations

import torch
import pytest

from hotpot_mas.training.grpo_loss import (
    approx_kl_per_report,
    batch_loss,
    batch_objective,
    clip_fraction_per_report,
    clipped_token_objective,
    entropy_per_report,
    per_report_objectives,
    token_ratio,
)


class TestRatioAndClip:
    def test_ratio_is_exp_of_logp_difference(self):
        ratio = token_ratio(
            torch.tensor([[0.0]]), torch.tensor([[-0.5]])
        )
        assert ratio.item() == pytest.approx(torch.exp(torch.tensor(0.5)).item())

    def test_clip_caps_positive_ratio(self):
        # ratio 1.5 with advantage 1.0: min(1.5, 1.2) = 1.2
        obj = clipped_token_objective(
            torch.tensor([1.5]), torch.tensor([1.0]), 0.2
        )
        assert obj.item() == pytest.approx(1.2)

    def test_clip_leaves_ratio_inside_range(self):
        obj = clipped_token_objective(
            torch.tensor([1.1]), torch.tensor([1.0]), 0.2
        )
        assert obj.item() == pytest.approx(1.1)

    def test_negative_advantage_uses_max(self):
        # ratio 0.5, advantage -1: min(-0.5, -0.8) = -0.8
        obj = clipped_token_objective(
            torch.tensor([0.5]), torch.tensor([-1.0]), 0.2
        )
        assert obj.item() == pytest.approx(-0.8)


class TestPerReportObjectives:
    def test_mask_limits_loss_to_report_tokens(self):
        # Token 0 masked out with a wildly different logp: the objective
        # must depend only on token 1.
        logp_new = torch.tensor([[100.0, -0.1]], requires_grad=True)
        logp_old = torch.tensor([[-100.0, -0.2]])
        mask = torch.tensor([[0.0, 1.0]])
        objective = per_report_objectives(
            logp_new, logp_old, torch.tensor([2.0]), mask, 0.2
        )
        # ratio on token 1 = exp(0.1) ~ 1.105, clipped at 1.2 -> 1.105 * 2
        expected = torch.exp(torch.tensor(0.1)).item() * 2.0
        assert objective.item() == pytest.approx(expected)

    def test_objective_averages_over_report_length(self):
        # Two reports with different lengths: each objective is its own
        # per-token mean, so a constant ratio gives the same objective.
        logp_new = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.0]])
        logp_old = torch.zeros_like(logp_new)
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0]])
        objectives = per_report_objectives(
            logp_new, logp_old, torch.tensor([1.0, 1.0]), mask, 0.2
        )
        assert objectives[0].item() == pytest.approx(
            objectives[1].item()
        )

    def test_empty_report_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            per_report_objectives(
                torch.tensor([[0.1]]),
                torch.tensor([[0.0]]),
                torch.tensor([1.0]),
                torch.tensor([[0.0]]),
                0.2,
            )

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            per_report_objectives(
                torch.tensor([[0.1]]),
                torch.tensor([[0.0], [0.0]]),
                torch.tensor([1.0]),
                torch.tensor([[1.0]]),
                0.2,
            )


class TestBatchWeighting:
    def test_batch_objective_is_weighted_sum(self):
        per_report = torch.tensor([0.5, 0.25])
        # One question with |S_q| = 2: weight 1/(1 * 2) each.
        weights = torch.tensor([0.5, 0.5])
        assert batch_objective(per_report, weights).item() == pytest.approx(
            0.375
        )
        assert batch_loss(per_report, weights).item() == pytest.approx(-0.375)

    def test_mixed_sq_weighting(self):
        # Question 1 contributes 2 reports, question 2 contributes 4:
        # weights 1/(2*2) and 1/(2*4) with |Q| = 2.
        per_report = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        weights = torch.tensor([0.25, 0.25, 0.125, 0.125, 0.125, 0.125])
        assert batch_objective(per_report, weights).item() == pytest.approx(
            1.0
        )


class TestDiagnostics:
    def test_approx_kl_is_mean_old_minus_new(self):
        logp_new = torch.tensor([[0.3, 0.7]])
        logp_old = torch.tensor([[0.1, 0.5]])
        mask = torch.tensor([[1.0, 1.0]])
        kl = approx_kl_per_report(logp_new, logp_old, mask)
        assert kl.item() == pytest.approx((-0.2 + -0.2) / 2)

    def test_approx_kl_ignores_masked_tokens(self):
        logp_new = torch.tensor([[100.0, 0.7]])
        logp_old = torch.tensor([[-100.0, 0.5]])
        mask = torch.tensor([[0.0, 1.0]])
        kl = approx_kl_per_report(logp_new, logp_old, mask)
        assert kl.item() == pytest.approx(0.5 - 0.7)

    def test_clip_fraction_counts_out_of_range_ratios(self):
        logp_new = torch.tensor([[0.0, 0.5, 0.0]])
        logp_old = torch.zeros_like(logp_new)
        mask = torch.tensor([[1.0, 1.0, 1.0]])
        # ratios: exp(0)=1 (in range), exp(0.5)~1.649 (out), 1 (in)
        frac = clip_fraction_per_report(logp_new, logp_old, mask, 0.2)
        assert frac.item() == pytest.approx(1.0 / 3.0)

    def test_entropy_is_negative_mean_logp(self):
        logp_new = torch.tensor([[-0.5, -1.5]])
        mask = torch.tensor([[1.0, 1.0]])
        entropy = entropy_per_report(logp_new, mask)
        assert entropy.item() == pytest.approx(1.0)

    def test_gradient_flows_through_objective(self):
        logp_new = torch.tensor([[0.0, 0.0]], requires_grad=True)
        logp_old = torch.tensor([[0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0]])
        loss = batch_loss(
            per_report_objectives(
                logp_new, logp_old, torch.tensor([1.0]), mask, 0.2
            ),
            torch.tensor([1.0]),
        )
        loss.backward()
        assert logp_new.grad is not None
        assert torch.isfinite(logp_new.grad).all()
