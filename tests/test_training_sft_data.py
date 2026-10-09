"""CPU-only tests for Stage 2 SFT example construction (sft_data.py).

The gold-answer-only supervision, wrapper masking, causal alignment,
padding, and the prefix-stability guards are all exercised offline with
the ``FakeChatTokenizer`` double.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from hotpot_mas.training.fake_policy import FakeChatTokenizer
from hotpot_mas.training.sft_data import (
    C_WRAPPER_FORMAT,
    C_WRAPPER_PREFIX,
    C_WRAPPER_SUFFIX,
    IGNORE_INDEX,
    build_sft_example,
    collate_sft_examples,
)

GOLD = "gold answer text"


def make_messages(
    question: str = "Question text?",
    a_report: str = "banana",
    b_report: str = "banana",
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "Answer from the two reports."},
        {
            "role": "user",
            "content": (
                f"Original question:\n{question}\n\n"
                f"Report from Alice:\n{a_report}\n\n"
                f"Report from Bob:\n{b_report}"
            ),
        },
    ]


def make_example(
    gold: str = GOLD, tokenizer: Any = None
) -> Any:
    tokenizer = tokenizer or FakeChatTokenizer()
    return build_sft_example(tokenizer, make_messages(), gold)


def test_wrapper_format_constant() -> None:
    assert C_WRAPPER_PREFIX == "Celab: <FINAL>"
    assert C_WRAPPER_SUFFIX == "</FINAL>"
    assert C_WRAPPER_FORMAT.format("x") == "Celab: <FINAL>x</FINAL>"


def test_only_gold_tokens_supervised() -> None:
    example = make_example()
    assert example.supervised_tokens == 3  # "gold", "answer", "text"
    assert len(example.input_ids) == len(example.labels)
    supervised_positions = [
        t for t, label in enumerate(example.labels) if label != IGNORE_INDEX
    ]
    assert len(supervised_positions) == 3
    # The supervised tokens are exactly the gold words, at the positions
    # where the input contains them.
    supervised_tokens = [example.input_ids[t] for t in supervised_positions]
    tokenizer = FakeChatTokenizer()
    assert supervised_tokens == [
        tokenizer._word_to_id[word] for word in ("gold", "answer", "text")
    ]
    # The labels store the true next-token ids on supervised positions.
    for t in supervised_positions:
        assert example.labels[t] == example.input_ids[t]
    # Prompt tokens (question + reports + generation prompt) are masked.
    assert all(
        label == IGNORE_INDEX for label in example.labels[: example.prompt_len]
    )
    # The wrapper "Celab: <FINAL>" and closing "</FINAL>" tokens are masked:
    # everything after the 3 gold tokens inside the completion is -100.
    completion_labels = example.labels[example.prompt_len :]
    assert completion_labels.count(IGNORE_INDEX) == (
        len(completion_labels) - 3
    )


def test_prompt_contains_no_gold() -> None:
    tokenizer = FakeChatTokenizer()
    messages = make_messages(a_report="banana", b_report="banana")
    prompt_str = str(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    )
    assert GOLD not in prompt_str
    example = build_sft_example(tokenizer, messages, GOLD)
    # No supervised token inside the prompt region.
    assert all(
        label == IGNORE_INDEX for label in example.labels[: example.prompt_len]
    )


def test_supervised_span_never_in_prompt() -> None:
    example = make_example()
    supervised = [
        t
        for t, label in enumerate(example.labels)
        if label != IGNORE_INDEX
    ]
    assert all(t >= example.prompt_len for t in supervised)


def test_variable_length_padding() -> None:
    tokenizer = FakeChatTokenizer()
    short = build_sft_example(
        tokenizer, make_messages(a_report="x"), GOLD
    )
    long = build_sft_example(
        tokenizer,
        make_messages(a_report="x " * 20, b_report="y " * 15),
        GOLD,
    )
    assert len(long.input_ids) > len(short.input_ids)
    input_ids, attention_mask, labels = collate_sft_examples(
        [short, long], pad_token_id=tokenizer.pad_token_id
    )
    assert input_ids.shape == labels.shape == (2, len(long.input_ids))
    assert attention_mask.shape == (2, len(long.input_ids))
    # Short example: padded positions carry pad ids, zero attention, and
    # ignored labels; the unpadded prefix is preserved.
    short_len = len(short.input_ids)
    assert list(input_ids[0, :short_len]) == short.input_ids
    assert input_ids[0, short_len:].eq(tokenizer.pad_token_id).all()
    assert attention_mask[0, :short_len].eq(1).all()
    assert attention_mask[0, short_len:].eq(0).all()
    assert labels[0, :short_len].tolist() == short.labels
    assert labels[0, short_len:].eq(IGNORE_INDEX).all()
    # Long example: fully unmasked.
    assert list(input_ids[1]) == long.input_ids
    assert attention_mask[1].eq(1).all()
    assert labels[1].tolist() == long.labels


def test_prefix_instability_raises() -> None:
    class UnstableTokenizer(FakeChatTokenizer):
        """Rendering that rewrites the prompt when a turn is appended."""

        def apply_chat_template(
            self,
            messages: List[Dict[str, str]],
            tokenize: bool = True,
            add_generation_prompt: bool = False,
        ) -> Any:
            rendered = super().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if any(m.get("role") == "assistant" for m in messages):
                rendered = "[REWRITTEN] " + rendered
            if tokenize:
                return self._tokenize(rendered)[0]
            return rendered

    with pytest.raises(RuntimeError, match="not prefix-stable"):
        build_sft_example(UnstableTokenizer(), make_messages(), GOLD)


def test_zero_supervised_raises() -> None:
    class WholeStringTokenizer(FakeChatTokenizer):
        """One token per string: no token lies fully inside the answer."""

        def _tokenize(self, text: str) -> Any:
            if not text:
                return [], []
            return [self.unk_token_id], [(0, len(text))]

    with pytest.raises(RuntimeError, match="cannot isolate the answer"):
        build_sft_example(WholeStringTokenizer(), make_messages(), GOLD)


def test_empty_gold_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_sft_example(FakeChatTokenizer(), make_messages(), "   ")


def test_wrapper_gold_alignment_checked() -> None:
    class CorruptingTokenizer(FakeChatTokenizer):
        """Renders the wrapper correctly but shifts the gold characters."""

        def apply_chat_template(
            self,
            messages: List[Dict[str, str]],
            tokenize: bool = True,
            add_generation_prompt: bool = False,
        ) -> Any:
            rendered = super().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if any(m.get("role") == "assistant" for m in messages):
                # Drop one character of the gold so the span checks fail.
                at = rendered.find(C_WRAPPER_PREFIX)
                if at >= 0:
                    rendered = rendered[: at + len(C_WRAPPER_PREFIX) + 1] + \
                        rendered[at + len(C_WRAPPER_PREFIX) + 2 :]
            if tokenize:
                return self._tokenize(rendered)[0]
            return rendered

    with pytest.raises(RuntimeError, match="gold answer"):
        build_sft_example(CorruptingTokenizer(), make_messages(), GOLD)
