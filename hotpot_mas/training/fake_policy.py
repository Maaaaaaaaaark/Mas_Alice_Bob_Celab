"""Offline test doubles: FakeTokenizer, FakePolicy, FakeSynthesizer.

``FakePolicy`` is a context-free categorical policy over a tiny vocabulary:
the trainable parameters are a single ``logits`` tensor ``theta`` of shape
``[vocab]``, and per-token log probs are ``log_softmax(theta)[token]``.
Sampling is a real seeded ``torch.multinomial`` and teacher forcing goes
through real autograd, so the exact GRPO loss / ratio / clip / approx-KL /
entropy code paths are exercised by the CPU-only unit tests without any
model download.

Because the policy is a single shared parameter tensor, giving the same
``FakePolicy`` instance to both workers A and B makes the "A/B share one
LoRA" property trivially observable (and assertable) in tests.

``FakeSynthesizer`` plays the frozen C: a scripted ``answer_fn(question,
a_text, b_text) -> str`` with a call counter, so tests can script exact
reward matrices and assert that C is called exactly G*G times per question.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from hotpot_mas.evaluation import token_overlap_scores
from hotpot_mas.parser import extract_final_answer
from hotpot_mas.seeds import seed_torch

from .config import DecodeConfig
from .policy import Report, _behavior_log_probs
from .synthesizers import SynthResult


class FakeTokenizer:
    """Deterministic whitespace tokenizer over a fixed tiny vocabulary."""

    def __init__(self, vocab: Optional[List[str]] = None):
        self.vocab = list(vocab) if vocab else [f"w{i}" for i in range(8)]
        self.eos_token = "[eos]"
        self.vocab.append(self.eos_token)
        self.eos_token_id = len(self.vocab) - 1
        self._word_to_id = {w: i for i, w in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        return [
            self._word_to_id.get(word, 0) for word in text.split()
        ]

    def decode(self, token_ids: List[int]) -> str:
        return " ".join(
            self.vocab[i] if 0 <= i < len(self.vocab) else "[?]"
            for i in token_ids
        )


class FakePolicy:
    """Categorical ``theta`` policy implementing the ``Policy`` protocol."""

    def __init__(
        self,
        tokenizer: Optional[FakeTokenizer] = None,
        init_logits: Optional[Tensor] = None,
    ):
        self.tokenizer = tokenizer or FakeTokenizer()
        vocab_size = len(self.tokenizer.vocab)
        if init_logits is None:
            self.logits = nn.Parameter(torch.zeros(vocab_size))
        else:
            if list(init_logits.shape) != [vocab_size]:
                raise ValueError(
                    f"init_logits must have shape [{vocab_size}]"
                )
            self.logits = nn.Parameter(init_logits.clone())
        self._calls: List[str] = []

    # -- tokenizer surface ----------------------------------------------

    def tokenize(self, messages: List[Dict[str, str]]) -> List[int]:
        text = " ".join(str(m.get("content", "")) for m in messages)
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    # -- sampling -------------------------------------------------------

    def _log_probs(self) -> Tensor:
        return torch.log_softmax(self.logits, dim=0)

    def _sample_loop(
        self,
        generation_seed: int,
        decode: DecodeConfig,
    ) -> Report:
        seed_torch(generation_seed)
        ids: List[int] = []
        logprobs: List[float] = []
        finish_reason = "length"
        with torch.no_grad():
            log_probs = _behavior_log_probs(self.logits, decode)
            for _ in range(decode.max_new_tokens):
                if decode.do_sample:
                    next_id = int(
                        torch.multinomial(log_probs.exp(), 1).item()
                    )
                else:
                    next_id = int(torch.argmax(log_probs).item())
                ids.append(next_id)
                logprobs.append(float(log_probs[next_id].item()))
                if next_id == self.tokenizer.eos_token_id:
                    finish_reason = "eos"
                    break
        return Report(
            token_ids=ids,
            text=self.decode(ids),
            logprobs=logprobs,
            finish_reason=finish_reason,
        )

    def sample_report(
        self,
        messages: List[Dict[str, str]],
        generation_seed: int,
        decode: DecodeConfig,
    ) -> Report:
        self._calls.append("sample")
        return self._sample_loop(generation_seed, decode)

    def decode_report(
        self,
        messages: List[Dict[str, str]],
        generation_seed: int,
        decode: DecodeConfig,
    ) -> Report:
        self._calls.append("decode")
        return self._sample_loop(generation_seed, decode)

    def teacher_force(
        self,
        input_ids: List[int],
        completion_ids: List[int],
        decode: Optional[DecodeConfig] = None,
    ) -> Tensor:
        """Per-token log probs under the current ``theta`` (grad enabled)."""
        if not completion_ids:
            raise ValueError("completion_ids must be non-empty")
        log_probs = _behavior_log_probs(
            self.logits, decode or DecodeConfig()
        )
        index = torch.tensor(
            completion_ids, device=self.logits.device, dtype=torch.long
        )
        return log_probs[index]

    # -- parameter bookkeeping ------------------------------------------

    def trainable_parameters(self) -> List[Any]:
        return [self.logits]

    def save_adapter(self, adapter_dir: Any) -> None:
        """Persist theta in the same directory layout as HFPolicy."""
        from pathlib import Path

        Path(adapter_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(adapter_dir) / "fake_adapter.pt")

    def load_adapter(self, adapter_dir: Any) -> None:
        from pathlib import Path

        path = Path(adapter_dir) / "fake_adapter.pt"
        if not path.is_file():
            raise FileNotFoundError(f"fake adapter not found: {path}")
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state)

    def state_dict(self) -> Dict[str, Tensor]:
        return {"logits": self.logits.detach().clone()}

    def load_state_dict(self, state: Dict[str, Tensor]) -> None:
        if set(state) != {"logits"}:
            raise ValueError("FakePolicy state must contain only 'logits'")
        with torch.no_grad():
            self.logits.copy_(state["logits"].to(
                self.logits.device, self.logits.dtype
            ))

    def parameter_norm_summary(self) -> Dict[str, Any]:
        l2 = float(self.logits.detach().float().norm().item())
        return {
            "trainable_parameters": 1,
            "trainable_numel": self.logits.numel(),
            "total_l2_norm": l2,
            "per_parameter_l2": {"logits": l2},
        }

    def info(self) -> Dict[str, Any]:
        return {
            "policy_kind": "FakePolicy",
            "vocab_size": len(self.tokenizer.vocab),
            "device": str(self.logits.device),
        }


class FakeSynthesizer:
    """Scripted stand-in for the frozen synthesizer C.

    ``answer_fn(question, a_text, b_text) -> str`` produces the raw answer
    text; the same parsing and official HotpotQA token-F1 code used by the
    real ``HFSynthesizer`` scores it against the gold answer. Every call is
    recorded as ``(question, a_report, b_report)`` so tests can assert the
    exact G*G call count and the i-major order.
    """

    def __init__(
        self,
        answer_fn: Callable[[str, str, str], str],
    ):
        self.answer_fn = answer_fn
        self.calls: List[Tuple[str, str, str]] = []

    def answer(
        self,
        messages: List[Dict[str, str]],
        gold: str,
        question: str = "",
        a_report: str = "",
        b_report: str = "",
    ) -> SynthResult:
        self.calls.append((question, a_report, b_report))
        raw_output = self.answer_fn(question, a_report, b_report)
        parsed = extract_final_answer(raw_output)
        if parsed.status == "ok" and parsed.body is not None:
            pred_answer = parsed.body
            parsed_ok = True
        else:
            pred_answer = raw_output
            parsed_ok = False
        precision, recall, f1 = token_overlap_scores(pred_answer, gold)
        return SynthResult(
            question=question,
            a_report=a_report,
            b_report=b_report,
            gold=gold,
            raw_output=raw_output,
            pred_answer=pred_answer,
            parsed=parsed_ok,
            precision=precision,
            recall=recall,
            f1=f1,
            input_tokens=0,  # the fake has no real tokenizer
            generated_tokens=0,
            finish_reason="eos",
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def info(self) -> Dict[str, Any]:
        return {"synthesizer_kind": "FakeSynthesizer", "frozen": True}
