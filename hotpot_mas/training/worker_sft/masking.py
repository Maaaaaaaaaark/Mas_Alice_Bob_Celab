"""SFT example masking and the causal loss for Stage 1a.

Difference from Stage 2 (deliberate): the Stage 2 synthesizer SFT masks
everything except a gold answer inside Celab/<FINAL> wrappers. This
worker task does NOT copy that. The worker is supervised on its ENTIRE
assistant turn — every token from the first target token through the
model's end-of-turn token — because the worker must learn the output
format (``D0: "..."`` / ``NOT FOUND``), the document ids, the exact
quotation text, and where the turn ends.

Masking (``-100`` = ignored):

- system/user prompt tokens,
- padding tokens,
- any non-target chat-template tokens (none in the assistant turn here;
  the template's assistant prefix is prompt-side and masked).

Causal alignment: a causal LM's logits at position ``t`` predict the
token at position ``t + 1``, so the loss aligns ``logits[:, :-1]`` with
``labels[:, 1:]``. With the whole assistant turn labeled, the first
target token is predicted from the last prompt position and the
end-of-turn token is predicted from the last content token; the code
asserts both are trained.

Loss (as documented in ``docs/WORKER_SFT.md``): for one example with
supervised positions ``P``, ``L = (1/|P|) * sum_{p in P}
-log P(correct token at p | prefix)``. The batch/update loss is the mean
of the example losses. ``accumulate_worker_sft_losses`` exists so the
training loop can sum per-example losses across microbatches and divide
by the number of examples in the group — this keeps every example's
weight the same for the final partial accumulation group, which plain
loss averaging would not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

IGNORE_INDEX = -100


@dataclass
class WorkerSFTExample:
    """One tokenized example with its -100 mask."""

    input_ids: Tensor  # [T]
    labels: Tensor  # [T], -100 outside the assistant turn
    prompt_len: int  # tokens before the first target token
    target_token_ids: Tensor  # [n] the supervised assistant tokens
    eos_token_id: int
    eos_position: int  # position of the end-of-turn token
    readable_tokens: List[str]

    def to_trace_dict(self) -> Dict[str, Any]:
        return {
            "input_len": int(self.input_ids.shape[0]),
            "prompt_len": self.prompt_len,
            "target_len": int(self.target_token_ids.shape[0]),
            "eos_token_id": self.eos_token_id,
            "eos_position": self.eos_position,
        }


def build_worker_sft_example(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    target: str,
) -> WorkerSFTExample:
    """Tokenize prompt + assistant target and build the -100 mask.

    The prompt and the full conversation are each rendered with the
    model's native chat template; the full conversation must extend the
    prompt exactly (string level AND token-id level), otherwise the
    template does not support prefix construction and the example
    cannot be supervised this way.
    """
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_str = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": target}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_str.startswith(prompt_str):
        raise ValueError(
            "chat template does not extend the generation prompt with the "
            "assistant turn: prefix construction failed at string level"
        )
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_str, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "chat template tokenization does not preserve the prompt prefix"
        )
    prompt_len = len(prompt_ids)
    if len(full_ids) <= prompt_len:
        raise ValueError("assistant turn tokenized to zero tokens")

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer has no eos_token_id")
    if int(full_ids[-1]) != int(eos_token_id):
        raise ValueError(
            "the assistant turn does not end with the model's end-of-turn "
            f"token (expected {eos_token_id}, got {int(full_ids[-1])})"
        )

    labels = torch.full(
        (len(full_ids),), IGNORE_INDEX, dtype=torch.long
    )
    labels[prompt_len:] = torch.tensor(full_ids[prompt_len:], dtype=torch.long)

    readable_tokens = []
    for token_id in full_ids.tolist():
        try:
            readable = tokenizer.decode([token_id])
        except Exception:  # noqa: BLE001 - decode is best-effort
            readable = f"<id {token_id}>"
        readable_tokens.append(readable)

    return WorkerSFTExample(
        input_ids=torch.tensor(full_ids, dtype=torch.long),
        labels=labels,
        prompt_len=prompt_len,
        target_token_ids=torch.tensor(
            full_ids[prompt_len:], dtype=torch.long
        ),
        eos_token_id=int(eos_token_id),
        eos_position=len(full_ids) - 1,
        readable_tokens=readable_tokens,
    )


def per_position_nll(logits: Tensor, labels: Tensor) -> Tensor:
    """Per-position negative log likelihood, shape ``[B, T-1]``.

    Entry ``[b, t]`` is the NLL of ``labels[b, t+1]`` under
    ``logits[b, t]`` (causal shift), or exactly 0.0 where the shifted
    label is ignored. Non-finite logits at ignored positions never leak
    through the ``where``.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, T, V], got {tuple(logits.shape)}")
    batch, time_steps, _ = logits.shape
    if labels.shape != (batch, time_steps):
        raise ValueError(f"labels must be [B, T], got {tuple(labels.shape)}")
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.clamp(min=0)
    nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = shift_labels != IGNORE_INDEX
    return torch.where(mask, nll, torch.zeros_like(nll))


def per_example_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean NLL over each example's own supervised tokens, shape ``[B]``."""
    nll = per_position_nll(logits, labels)
    counts = (labels[:, 1:] != IGNORE_INDEX).sum(dim=1).clamp(min=1).float()
    return nll.sum(dim=1) / counts


def accumulate_worker_sft_losses(
    losses: List[Tensor], counts: List[int]
) -> Tensor:
    """Weighted update loss from per-example losses.

    ``losses`` holds one per-example loss per microbatch;
    ``counts[i]`` is the number of examples the i-th entry represents.
    The update loss is ``sum(losses[i] * counts[i]) / sum(counts)``, so
    every example contributes its own per-example loss with equal weight
    regardless of how the accumulation group is split — the final
    partial group keeps the same per-example weighting.
    """
    total = sum(int(c) for c in counts)
    if total <= 0:
        raise ValueError("no examples accumulated")
    weighted = sum(
        loss * count
        for loss, count in zip(losses, counts)
        if count > 0
    )
    return weighted / total


def supervised_token_records(
    sft_example: WorkerSFTExample,
    position_nlls: Tensor,
) -> List[Dict[str, Any]]:
    """Per-supervised-token records for the trace (section G).

    ``position_nlls`` is the ``[T-1]`` per-position NLL row of THIS
    example's forward pass. Each supervised target token at position
    ``p`` was predicted from logits at ``p-1``.
    """
    records = []
    for offset, token_id in enumerate(sft_example.target_token_ids.tolist()):
        position = sft_example.prompt_len + offset
        prediction_position = position - 1
        nll = float(position_nlls[prediction_position])
        readable = (
            sft_example.readable_tokens[position]
            if position < len(sft_example.readable_tokens)
            else f"<id {token_id}>"
        )
        records.append(
            {
                "position": position,
                "prediction_position": prediction_position,
                "token_id": token_id,
                "token": readable,
                "nll": nll,
                "log_prob": -nll,
                "contributes_to_loss": True,
            }
        )
    return records


def assert_supervised_bounds(
    sft_example: WorkerSFTExample, max_input_tokens: int
) -> None:
    """Fail loudly instead of silently truncating evidence or target."""
    total = int(sft_example.input_ids.shape[0])
    if total > max_input_tokens:
        raise ValueError(
            f"example exceeds max_input_tokens={max_input_tokens}: "
            f"prompt_len={sft_example.prompt_len}, total={total}. "
            "The target or visible documents would be truncated; raise "
            "max_input_tokens after inspecting the lengths report, never "
            "silently truncate."
        )
