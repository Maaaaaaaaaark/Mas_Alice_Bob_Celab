"""Pure SFT loss functions (Eq.~(4) of ``cross_paired_grpo.tex``).

Causal shift: a causal LM's logits at position ``t`` predict the token at
position ``t + 1``, so the loss aligns ``logits[:, :-1]`` with
``labels[:, 1:]``. Only positions whose shifted label is not the ignore
index contribute. Each example is averaged over its own gold-answer tokens
(the TeX's ``1/|y*|`` weighting) and the batch loss is the mean over
examples — the TeX weighting, not HuggingFace's token-averaged default.

The functions are pure tensor code so the exact loss path runs unchanged
in CPU-only unit tests and inside the GPU trainer.
"""

from __future__ import annotations

import math

from torch import Tensor
import torch

from .sft_data import IGNORE_INDEX


def shifted_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Per-position negative log likelihood, shape ``[B, T - 1]``.

    Entry ``[b, t]`` is the NLL of ``labels[b, t + 1]`` under
    ``logits[b, t]``, or ``0.0`` where the shifted label is ignored.
    """
    if logits.dim() != 3:
        raise ValueError(
            f"logits must be [B, T, V], got {tuple(logits.shape)}"
        )
    batch, time_steps, _ = logits.shape
    if labels.shape != (batch, time_steps):
        raise ValueError(f"labels must be [B, T], got {tuple(labels.shape)}")
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    # IGNORE_INDEX is negative; clamp to a valid index for the gather and
    # zero the result with the mask afterwards. ``where`` (not ``* 0``)
    # keeps the masked positions exactly zero even when the model emits
    # non-finite logits at padded positions.
    safe_labels = shift_labels.clamp(min=0)
    nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = shift_labels != IGNORE_INDEX
    return torch.where(mask, nll, torch.zeros_like(nll))


def per_example_sft_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean NLL over each example's own gold-answer tokens, shape ``[B]``."""
    nll = shifted_cross_entropy(logits, labels)
    counts = (labels[:, 1:] != IGNORE_INDEX).sum(dim=1).clamp(min=1).float()
    return nll.sum(dim=1) / counts


def batch_sft_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Scalar SFT loss: mean over examples of their per-token NLL."""
    return per_example_sft_loss(logits, labels).mean()


def assert_finite(value: Tensor, what: str) -> None:
    """Raise on any NaN/inf in a loss or diagnostic tensor."""
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{what} contains non-finite values")


def assert_finite_float(value: float, what: str) -> None:
    """Raise on a non-finite scalar (NaN/inf) before it is recorded."""
    if not math.isfinite(value):
        raise FloatingPointError(f"{what} is not finite: {value!r}")
