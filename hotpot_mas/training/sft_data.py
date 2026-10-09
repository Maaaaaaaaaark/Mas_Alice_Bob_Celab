"""Stage 2 SFT example construction (Algorithm 2 of ``cross_paired_grpo.tex``).

One example per training question: the input is the C chat prompt (question
+ Alice report + Bob report, exactly the messages used at evaluation)
followed by the assistant completion ``Celab: <FINAL>{gold}</FINAL>``; the
labels are ``-100`` everywhere except on the gold-answer tokens. Only the
gold-answer tokens participate in the loss (Eq.~(4) of the TeX); the
question, both reports and the wrapper are conditioning only.

The answer span is located by rendering the full chat string with the
tokenizer's chat template, re-encoding it with character offsets, and
mapping the gold-answer character span onto tokens — never by assuming that
separately tokenized pieces concatenate to the in-context tokenization.

Two prefix-stability properties are verified at build time and the
constructor raises a clear error if a tokenizer/chat template violates
them, instead of silently training on a misaligned span:

1. appending the assistant turn must not change the rendered string prefix
   (``full_str`` starts with ``prompt_str``);
2. appending the assistant turn must not change the tokenization of the
   prompt (``full_ids[:len(prompt_ids)] == prompt_ids``), and re-encoding
   the rendered string must reproduce the chat-template token ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

C_WRAPPER_PREFIX = "Celab: <FINAL>"
C_WRAPPER_SUFFIX = "</FINAL>"
C_WRAPPER_FORMAT = f"{C_WRAPPER_PREFIX}{{}}{C_WRAPPER_SUFFIX}"

IGNORE_INDEX = -100


@dataclass
class SFTExample:
    """One D_C example (q, a, b, y*) with the label mask resolved."""

    question_id: str
    input_ids: List[int]  # prompt + completion (teacher-forcing input)
    labels: List[int]  # IGNORE_INDEX except on the gold-answer tokens
    prompt_len: int  # token count of the masked prompt (question + reports)
    completion_len: int  # token count of the assistant completion
    gold_answer: str
    supervised_tokens: int  # gold-answer token count inside labels

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.labels):
            raise ValueError("input_ids and labels must have equal length")
        if self.prompt_len + self.completion_len != len(self.input_ids):
            raise ValueError("prompt_len + completion_len must equal length")


def _tokenize_with_offsets(
    tokenizer: Any, text: str
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Encode ``text`` and return (ids, per-token character offsets)."""
    enc = tokenizer(
        text, return_offsets_mapping=True, add_special_tokens=False
    )
    ids = [int(i) for i in enc["input_ids"]]
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
    return ids, offsets


def build_sft_example(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    gold: str,
    question_id: str = "",
) -> SFTExample:
    """Build one SFT example with labels restricted to the gold answer.

    ``tokenizer`` must expose ``apply_chat_template`` (with ``tokenize``
    and ``add_generation_prompt`` keywords) and the usual ``__call__`` with
    ``return_offsets_mapping`` — the HuggingFace tokenizer surface. The
    chat template is expected to render the assistant turn starting with
    the same generation-prompt suffix, which is checked below.
    """
    if not isinstance(gold, str) or not gold.strip():
        raise ValueError("gold answer must be a non-empty string")

    prompt_ids = [
        int(i)
        for i in tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    ]
    prompt_str = str(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    )

    assistant_content = C_WRAPPER_FORMAT.format(gold)
    full_messages = list(messages) + [
        {"role": "assistant", "content": assistant_content}
    ]
    full_ids = [
        int(i)
        for i in tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False
        )
    ]
    full_str = str(
        tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
    )

    # Prefix stability: the assistant turn must extend, never rewrite, the
    # prompt rendering and its tokenization.
    if not full_str.startswith(prompt_str):
        raise RuntimeError(
            "chat template is not prefix-stable in the rendered string: "
            "appending the assistant turn changed the prompt text; the "
            "gold-answer span cannot be located reliably"
        )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "chat template is not prefix-stable in tokenization: appending "
            "the assistant turn changed the prompt token ids; the "
            "gold-answer span cannot be located reliably"
        )

    ids, offsets = _tokenize_with_offsets(tokenizer, full_str)
    if ids != full_ids:
        raise RuntimeError(
            "tokenizer(text) disagrees with "
            "apply_chat_template(tokenize=True) for the rendered chat "
            "string; the gold-answer span cannot be located reliably"
        )

    # Locate the gold answer inside the assistant turn (after the prompt).
    wrapper_at = full_str.find(C_WRAPPER_PREFIX, len(prompt_str))
    if wrapper_at < 0:
        raise RuntimeError(
            "assistant wrapper 'Celab: <FINAL>' not found after the prompt "
            "in the rendered chat string"
        )
    gold_char_start = wrapper_at + len(C_WRAPPER_PREFIX)
    gold_char_end = gold_char_start + len(gold)
    if full_str[gold_char_start:gold_char_end] != gold:
        raise RuntimeError(
            "gold answer does not follow the assistant wrapper in the "
            "rendered chat string"
        )
    closing_at = gold_char_end
    if (
        full_str[closing_at:closing_at + len(C_WRAPPER_SUFFIX)]
        != C_WRAPPER_SUFFIX
    ):
        raise RuntimeError(
            "gold answer is not followed by the closing '</FINAL>' tag in "
            "the rendered chat string"
        )

    labels = [IGNORE_INDEX] * len(full_ids)
    supervised = 0
    for t in range(len(prompt_ids), len(full_ids)):
        start, end = offsets[t]
        # A token is supervised only when it lies fully inside the gold
        # span; boundary-straddling tokens stay masked (safe under any
        # tokenizer that merges across the tag/answer boundary).
        if start >= gold_char_start and end <= gold_char_end:
            labels[t] = full_ids[t]
            supervised += 1
    if supervised == 0:
        raise RuntimeError(
            "no token lies fully inside the gold-answer span; this "
            "tokenizer/chat template cannot isolate the answer tokens, so "
            "the answer-only SFT objective cannot be built (refusing to "
            "silently supervise the whole completion)"
        )

    return SFTExample(
        question_id=question_id,
        input_ids=full_ids,
        labels=labels,
        prompt_len=len(prompt_ids),
        completion_len=len(full_ids) - len(prompt_ids),
        gold_answer=gold,
        supervised_tokens=supervised,
    )


def collate_sft_examples(
    examples: List[SFTExample], pad_token_id: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pad a variable-length batch to (input_ids, attention_mask, labels).

    Padding positions carry the pad token id with a zero attention weight
    and label ``IGNORE_INDEX``, so padded regions never contribute to the
    loss regardless of the model's logits there.
    """
    if not examples:
        raise ValueError("cannot collate an empty batch")
    max_len = max(len(example.input_ids) for example in examples)
    batch_size = len(examples)
    input_ids = torch.full(
        (batch_size, max_len), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros(
        (batch_size, max_len), dtype=torch.long
    )
    labels = torch.full(
        (batch_size, max_len), IGNORE_INDEX, dtype=torch.long
    )
    for i, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[i, :length] = torch.tensor(
            example.input_ids, dtype=torch.long
        )
        attention_mask[i, :length] = 1
        labels[i, :length] = torch.tensor(example.labels, dtype=torch.long)
    return input_ids, attention_mask, labels
