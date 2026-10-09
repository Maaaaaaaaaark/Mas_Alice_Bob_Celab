"""CPU-only tests for the Stage 2 SFT loss (sft_loss.py).

Covers: manual cross-entropy equality, the causal next-token shift (no
off-by-one), per-example ``1/|y*|`` averaging, padding robustness (NaN
logits at padded positions are ignored), and the non-finite guards.
"""

from __future__ import annotations

import math

import pytest
import torch

from hotpot_mas.training.sft_data import IGNORE_INDEX
from hotpot_mas.training.sft_loss import (
    assert_finite,
    assert_finite_float,
    batch_sft_loss,
    per_example_sft_loss,
    shifted_cross_entropy,
)


def manual_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Independent reference: per-position NLL, -100 positions are 0."""
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    out = torch.zeros_like(shift_labels, dtype=log_probs.dtype)
    for b in range(shift_labels.shape[0]):
        for t in range(shift_labels.shape[1]):
            label = int(shift_labels[b, t])
            if label == IGNORE_INDEX:
                continue
            out[b, t] = -log_probs[b, t, label]
    return out


def test_matches_manual_cross_entropy() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 5)
    labels = torch.tensor(
        [
            [IGNORE_INDEX, 2, IGNORE_INDEX, 1, 0, IGNORE_INDEX],
            [IGNORE_INDEX, 3, 4, IGNORE_INDEX, IGNORE_INDEX, 2],
        ]
    )
    got = shifted_cross_entropy(logits, labels)
    expected = manual_nll(logits, labels)
    assert torch.allclose(got, expected, atol=1e-6)


def test_no_off_by_one() -> None:
    """logits[t] scores token t+1: the final wrapper position scores the
    first gold token; the first gold position's own logits are ignored."""
    vocab = 5
    gold = 3
    logits = torch.zeros(1, 4, vocab)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, gold, IGNORE_INDEX]])
    base_loss = batch_sft_loss(logits, labels)

    # Perturbing logits at the last masked position (t=1) changes the loss:
    # it is the position whose prediction is the first gold token.
    perturbed = logits.clone()
    perturbed[0, 1, 2] += 5.0
    assert batch_sft_loss(perturbed, labels).item() != pytest.approx(
        base_loss.item()
    )

    # Perturbing logits at the first gold position (t=2) changes nothing:
    # its predicted token (the last position) is masked.
    perturbed = logits.clone()
    perturbed[0, 2, 2] += 5.0
    assert batch_sft_loss(perturbed, labels).item() == pytest.approx(
        base_loss.item()
    )


def test_per_example_averaging_not_token_weighted() -> None:
    torch.manual_seed(1)
    vocab = 5
    # Example 0 supervises 1 token, example 1 supervises 3 tokens.
    logits = torch.randn(2, 4, vocab)
    labels = torch.tensor(
        [
            [IGNORE_INDEX, IGNORE_INDEX, 2, IGNORE_INDEX],
            [IGNORE_INDEX, 1, 3, 0],
        ]
    )
    per_example = per_example_sft_loss(logits, labels)
    assert per_example.shape == (2,)
    # Each example equals its own mean; the batch is the mean of the two
    # example means, NOT the mean over all 4 supervised tokens.
    nll = manual_nll(logits, labels)
    assert per_example[0].item() == pytest.approx(nll[0, 1].item())
    assert per_example[1].item() == pytest.approx(
        (nll[1, 0] + nll[1, 1] + nll[1, 2]).item() / 3
    )
    token_averaged = (nll.sum() / 4).item()
    batch = batch_sft_loss(logits, labels).item()
    assert batch == pytest.approx(per_example.mean().item())
    assert batch != pytest.approx(token_averaged)


def test_padded_positions_ignored_even_with_nan_logits() -> None:
    """NaN logits at padded positions must not poison the loss."""
    torch.manual_seed(2)
    vocab = 5
    logits = torch.randn(1, 4, vocab)
    labels = torch.tensor([[IGNORE_INDEX, 1, 2, IGNORE_INDEX]])
    unpadded = batch_sft_loss(logits, labels).item()

    # Pad the example with two extra positions whose labels are ignored
    # and whose logits are NaN.
    padded_logits = torch.cat(
        [logits, torch.full((1, 2, vocab), float("nan"))], dim=1
    )
    padded_labels = torch.cat(
        [labels, torch.full((1, 2), IGNORE_INDEX, dtype=labels.dtype)],
        dim=1,
    )
    padded = batch_sft_loss(padded_logits, padded_labels)
    assert torch.isfinite(padded)
    assert padded.item() == pytest.approx(unpadded)


def test_non_finite_loss_raises() -> None:
    vocab = 4
    logits = torch.full((1, 3, vocab), float("nan"))
    labels = torch.tensor([[IGNORE_INDEX, 1, IGNORE_INDEX]])
    with pytest.raises(FloatingPointError, match="non-finite"):
        assert_finite(batch_sft_loss(logits, labels), "SFT loss")
    with pytest.raises(FloatingPointError, match="non-finite"):
        assert_finite(
            torch.tensor([0.0, float("inf")]), "grad norm"
        )
    with pytest.raises(FloatingPointError, match="not finite"):
        assert_finite_float(float("nan"), "train loss")
    with pytest.raises(FloatingPointError, match="not finite"):
        assert_finite_float(math.inf, "grad norm")
    # Finite values pass through.
    assert_finite(torch.tensor([0.0, 1.0]), "x")
    assert_finite_float(0.5, "x")


def test_shape_validation() -> None:
    with pytest.raises(ValueError, match="must be \\[B, T, V\\]"):
        shifted_cross_entropy(torch.zeros(2, 3), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="labels must be \\[B, T\\]"):
        shifted_cross_entropy(
            torch.zeros(2, 3, 4), torch.zeros(3, 3)
        )
