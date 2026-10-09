"""Token-level GRPO objective and diagnostics (pure tensor functions).

Implements the PPO-style objective of Algorithm 1 in ``cross_paired_grpo.tex``
without any model or rollout code, so every quantity here can be checked
against hand-computed values in unit tests:

    rho_t   = pi_theta(o_t | x, o_<t) / pi_old(o_t | x, o_<t)
    ell_t   = min(rho_t * u, clip(rho_t, 1 - eps, 1 + eps) * u)
    J       = (1/|Q|) sum_q (1/|S_q|) sum_{(x, o, u) in S_q} (1/|o|) sum_t ell_t
    L       = -J

Only worker report tokens carry loss: prompt/question/document tokens are
conditioning only. There is deliberately **no KL penalty** in the loss; the
policy drift is recorded as the diagnostic ``approx_kl`` (mean of
``old_logp - new_logp`` over report tokens), exactly as the experiment
specification requires.

All functions take ``new_logps``/``old_logps`` of shape ``[B, T]`` and a
float mask of the same shape (1 on report tokens, 0 elsewhere); advantages
are shape ``[B]``. Per-report quantities are returned as ``[B]`` so the
trainer can average them with the exact TeX weighting ``1/(|Q| |S_q|)``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def token_ratio(logp_new: Tensor, logp_old: Tensor) -> Tensor:
    """``rho_t = exp(logp_new - logp_old)``, elementwise."""
    return (logp_new - logp_old).exp()


def clipped_token_objective(
    ratio: Tensor, advantage: Tensor, clip_epsilon: float
) -> Tensor:
    """``ell_t = min(rho_t * u, clip(rho_t, 1-eps, 1+eps) * u)``."""
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return torch.minimum(ratio * advantage, clipped * advantage)


def per_report_objectives(
    logp_new: Tensor,
    logp_old: Tensor,
    advantages: Tensor,
    mask: Tensor,
    clip_epsilon: float,
) -> Tensor:
    """Per-report objective ``(1/|o|) sum_t ell_t`` over report tokens.

    ``logp_new``/``logp_old``: ``[B, T]``; ``advantages``: ``[B]``;
    ``mask``: ``[B, T]`` float mask, 1 on report tokens only. Returns
    shape ``[B]``. Reports with no masked tokens are rejected loudly
    (the trainer filters zero-length reports before this point).
    """
    if logp_new.shape != logp_old.shape:
        raise ValueError("logp_new and logp_old shapes must match")
    if mask.shape != logp_new.shape:
        raise ValueError("mask must have the same shape as the logprobs")
    if advantages.shape != (logp_new.shape[0],):
        raise ValueError(
            "advantages must have shape [B] (one per report)"
        )
    token_counts = mask.sum(dim=1)
    if (token_counts <= 0).any():
        raise ValueError(
            "every report must contain at least one masked (report) token; "
            "zero-length reports are filtered during rollout"
        )
    ratio = token_ratio(logp_new, logp_old)
    per_token = clipped_token_objective(
        ratio, advantages.unsqueeze(1), clip_epsilon
    )
    return (per_token * mask).sum(dim=1) / token_counts


def batch_objective(
    per_report: Tensor, weights: Tensor
) -> Tensor:
    """Weighted batch objective ``J = sum_i weight_i * per_report_i``.

    The caller supplies weights already containing the ``1/(|Q| |S_q|)``
    factors (gradient accumulation across minibatches keeps J TeX-exact).
    """
    if per_report.shape != weights.shape:
        raise ValueError("per_report and weights shapes must match")
    return (per_report * weights).sum()


def batch_loss(
    per_report: Tensor, weights: Tensor
) -> Tensor:
    """``L = -J``."""
    return -batch_objective(per_report, weights)


def approx_kl_per_report(
    logp_new: Tensor, logp_old: Tensor, mask: Tensor
) -> Tensor:
    """Per-report approx KL: mean of ``old_logp - new_logp`` over report
    tokens (a drift diagnostic, not a loss term)."""
    counts = mask.sum(dim=1)
    if (counts <= 0).any():
        raise ValueError("every report must contain at least one report token")
    return ((logp_old - logp_new) * mask).sum(dim=1) / counts


def clip_fraction_per_report(
    logp_new: Tensor, logp_old: Tensor, mask: Tensor, clip_epsilon: float
) -> Tensor:
    """Per-report fraction of report tokens with ``|rho_t - 1| > eps``."""
    counts = mask.sum(dim=1)
    if (counts <= 0).any():
        raise ValueError("every report must contain at least one report token")
    ratio = token_ratio(logp_new, logp_old)
    clipped = ((ratio - 1.0).abs() > clip_epsilon).float() * mask
    return clipped.sum(dim=1) / counts


def entropy_per_report(logp_new: Tensor, mask: Tensor) -> Tensor:
    """Per-report mean entropy ``-(1/|o|) sum_t logp_t`` over report tokens."""
    counts = mask.sum(dim=1)
    if (counts <= 0).any():
        raise ValueError("every report must contain at least one report token")
    return (-logp_new * mask).sum(dim=1) / counts
