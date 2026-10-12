"""Masking, causal-alignment, and loss tests for Stage 1a. CPU-only,
using the fake tokenizer (no model checkpoint, no GPU)."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch

from hotpot_mas.training.worker_sft.fakes import FakeWorkerTokenizer
from hotpot_mas.training.worker_sft.masking import (
    IGNORE_INDEX,
    accumulate_worker_sft_losses,
    assert_supervised_bounds,
    build_worker_sft_example,
    per_example_loss,
    per_position_nll,
    supervised_token_records,
)

from worker_sft_test_utils import make_doc, make_evidence, make_example

MESSAGES = [
    {"role": "system", "content": "You are a search worker."},
    {"role": "user", "content": "Question: When?\n\nDocuments:\n\nD0: Title\nText."},
]
TARGET = 'D0: "Text."'


def _tokenizer() -> FakeWorkerTokenizer:
    return FakeWorkerTokenizer(
        corpus=["You are a search worker.", "When?", "Text.",
                "Documents:", "D0:", '"Text."', "NOT", "FOUND"]
    )


def _example() -> Any:
    doc = make_doc("d0", "The tower opened in 1889.")
    return make_example(
        "pos-1",
        "When did the tower open?",
        [doc],
        answer="1889",
        evidence=[make_evidence("d0", "The tower opened in 1889.")],
    )


def test_prompt_tokens_are_masked_and_target_tokens_are_supervised():
    tokenizer = _tokenizer()
    sft_example = build_worker_sft_example(tokenizer, MESSAGES, TARGET)
    labels = sft_example.labels.tolist()
    assert all(label == IGNORE_INDEX for label in labels[: sft_example.prompt_len])
    assert all(label != IGNORE_INDEX for label in labels[sft_example.prompt_len:])
    # The supervised span equals the assistant turn's token ids.
    assert labels[sft_example.prompt_len:] == sft_example.input_ids.tolist()[
        sft_example.prompt_len:
    ]


def test_first_target_token_and_eot_are_supervised():
    tokenizer = _tokenizer()
    sft_example = build_worker_sft_example(tokenizer, MESSAGES, TARGET)
    # First target token sits at prompt_len and is predicted from the
    # last prompt position (causal shift: logits[t] predict token t+1).
    assert sft_example.labels[sft_example.prompt_len].item() != IGNORE_INDEX
    # The end-of-turn token is the last token and IS supervised.
    assert sft_example.eos_position == len(sft_example.input_ids) - 1
    assert sft_example.input_ids[-1].item() == tokenizer.eos_token_id
    assert sft_example.labels[-1].item() == tokenizer.eos_token_id


def test_chat_turn_end_may_differ_from_tokenizer_eos():
    class TurnEndTokenizer(FakeWorkerTokenizer):
        @property
        def all_special_ids(self):
            return list(self.vocab.values())[:6]

        def _render(self, messages, add_generation_prompt):
            return super()._render(messages, add_generation_prompt).replace(
                "<eos>", "<assistant_end>"
            )

    tokenizer = TurnEndTokenizer()
    sft_example = build_worker_sft_example(tokenizer, MESSAGES, TARGET)
    assert sft_example.eos_token_id == tokenizer.vocab["<assistant_end>"]
    assert sft_example.eos_token_id != tokenizer.eos_token_id
    assert sft_example.labels[-1].item() == sft_example.eos_token_id


def test_prefix_stability_is_checked_at_string_and_token_level():
    class BrokenTokenizer(FakeWorkerTokenizer):
        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False,
                                return_tensors=None):
            text = self._render(messages, add_generation_prompt)
            if not tokenize:
                return text
            if add_generation_prompt:
                # Break the prefix: drop the trailing assistant marker.
                text = text[:-len("assistant: ")]
            ids = [self._id_of(t) for t in text.split()]
            if return_tensors == "pt":
                return torch.tensor([ids], dtype=torch.long)
            return ids

    broken = BrokenTokenizer()
    # String-level check still passes, token-level must fail loudly.
    with pytest.raises(ValueError, match="prefix"):
        build_worker_sft_example(broken, MESSAGES, TARGET)


def test_per_position_nll_matches_manual_log_softmax():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 11)
    labels = torch.tensor([[0, 3, 1, -100, 2]])
    nll = per_position_nll(logits, labels)  # [1, T-1]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    for t in range(4):
        if labels[0, t + 1].item() == -100:
            assert nll[0, t].item() == 0.0
        else:
            expected = -log_probs[0, t, labels[0, t + 1]].item()
            assert abs(nll[0, t].item() - expected) < 1e-5


def test_per_example_loss_is_mean_over_supervised_tokens():
    logits = torch.zeros(1, 4, 7)
    labels = torch.tensor([[1, -100, 2, 3]])
    loss = per_example_loss(logits, labels)
    assert loss.shape == (1,)
    # Two supervised positions (t=2 and t=3 -> predicted from t=1, t=2).
    assert loss.item() == pytest.approx(-torch.log(torch.tensor(1 / 7)).item())


def test_accumulate_worker_sft_losses_keeps_per_example_weighting():
    # Group of 3 examples split 2 + 1 (a partial accumulation group).
    losses = [torch.tensor(0.5), torch.tensor(1.0), torch.tensor(2.0)]
    update = accumulate_worker_sft_losses(losses[:2], [2]).item()
    assert update == pytest.approx((0.5 + 1.0) / 2)
    # Combined across a boundary: (0.5 + 1.0 + 2.0) / 3 either way.
    total = accumulate_worker_sft_losses(losses, [3]).item()
    partial = accumulate_worker_sft_losses(
        [torch.tensor(0.75), torch.tensor(2.0)], [2, 1]
    ).item()
    assert total == pytest.approx(3.5 / 3)
    assert partial == pytest.approx((0.75 * 2 + 2.0) / 3)


def test_accumulate_refuses_no_examples():
    with pytest.raises(ValueError, match="no examples"):
        accumulate_worker_sft_losses([], [])


def test_supervised_token_records_positions_match_target_tokens():
    tokenizer = _tokenizer()
    sft_example = build_worker_sft_example(tokenizer, MESSAGES, TARGET)
    nll_row = torch.zeros(len(sft_example.input_ids) - 1)
    records = supervised_token_records(sft_example, nll_row)
    assert len(records) == len(sft_example.target_token_ids)
    for offset, record in enumerate(records):
        position = sft_example.prompt_len + offset
        assert record["position"] == position
        assert record["prediction_position"] == position - 1
        assert record["token_id"] == int(
            sft_example.target_token_ids[offset]
        )
        assert record["contributes_to_loss"] is True


def test_assert_supervised_bounds_raises_instead_of_truncating():
    tokenizer = _tokenizer()
    sft_example = build_worker_sft_example(tokenizer, MESSAGES, TARGET)
    with pytest.raises(ValueError, match="never silently truncate"):
        assert_supervised_bounds(
            sft_example, max_input_tokens=len(sft_example.input_ids) - 1
        )
    # Exactly at the bound is fine.
    assert_supervised_bounds(
        sft_example, max_input_tokens=len(sft_example.input_ids)
    )


def test_chat_template_contract_full_extends_prompt():
    tokenizer = _tokenizer()
    prompt = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        MESSAGES + [{"role": "assistant", "content": TARGET}],
        tokenize=False,
        add_generation_prompt=False,
    )
    assert full.startswith(prompt)
    assert full.endswith("<eos>")
