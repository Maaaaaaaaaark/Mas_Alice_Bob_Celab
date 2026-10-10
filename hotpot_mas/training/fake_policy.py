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

``FakeChatTokenizer`` and ``FakeTrainableSynthesizer`` are the Stage 2
doubles: a chat-template tokenizer whose position-independent regex
tokenization satisfies ``build_sft_example``'s prefix-stability checks, and
a C with frozen base logits plus a trainable LoRA logits parameter, so the
SFT loss path runs offline with real autograd.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from hotpot_mas.evaluation import token_overlap_scores
from hotpot_mas.parser import extract_final_answer
from hotpot_mas.seeds import seed_torch

from .config import DecodeConfig
from .policy import Report, _behavior_log_probs
from .sft_data import C_WRAPPER_PREFIX, C_WRAPPER_SUFFIX
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
        force_completed_reports: bool = True,
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
        self.force_completed_reports = force_completed_reports

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
            finish_reason=(
                "eos" if self.force_completed_reports else finish_reason
            ),
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

    def token_log_distributions(
        self,
        input_ids: List[int],
        completion_ids: List[int],
        decode: Optional[DecodeConfig] = None,
    ) -> Tensor:
        if not completion_ids:
            raise ValueError("completion_ids must be non-empty")
        log_probs = _behavior_log_probs(
            self.logits.detach(), decode or DecodeConfig()
        )
        return log_probs.unsqueeze(0).repeat(len(completion_ids), 1)

    # -- parameter bookkeeping ------------------------------------------

    def trainable_parameters(self) -> List[Any]:
        return [self.logits] if self.logits.requires_grad else []

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


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


class FakeChatTokenizer:
    """Chat-template tokenizer double for the Stage 2 SFT code.

    Tokenization is a position-independent regex over the whole string
    (word runs and single punctuation characters; whitespace is skipped),
    so appending a turn can never re-tokenize the prefix — the exact
    property ``build_sft_example`` verifies. The chat template renders
    non-assistant contents joined by spaces, assistant turns as
    ``"<assistant> {content}"``, and the generation prompt as the trailing
    ``" <assistant>"`` that opens the assistant turn, so both the string
    and the token ids of the prompt are an exact prefix of the full chat.
    """

    def __init__(self, vocab: Optional[List[str]] = None):
        words = list(vocab) if vocab else ["gold", "answer", "text", "banana"]
        self.vocab = ["<pad>", "<unk>"] + words + ["<eos>"]
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = len(self.vocab) - 1
        self._word_to_id = {w: i for i, w in enumerate(self.vocab)}

    def _tokenize(
        self, text: str
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        ids: List[int] = []
        offsets: List[Tuple[int, int]] = []
        for match in _TOKEN_RE.finditer(text):
            word = match.group(0)
            ids.append(self._word_to_id.get(word, self.unk_token_id))
            offsets.append((match.start(), match.end()))
        return ids, offsets

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> Any:
        parts: List[str] = []
        for message in messages:
            content = str(message.get("content", ""))
            if message.get("role") == "assistant":
                parts.append("<assistant> " + content)
            else:
                parts.append(content)
        rendered = " ".join(parts)
        if add_generation_prompt:
            rendered += " <assistant>"
        if not tokenize:
            return rendered
        ids, _ = self._tokenize(rendered)
        return ids

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        ids, offsets = self._tokenize(str(text))
        out: Dict[str, Any] = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def decode(self, token_ids: List[int]) -> str:
        parts: List[str] = []
        for i in token_ids:
            if i in (self.pad_token_id, self.unk_token_id, self.eos_token_id):
                continue
            word = self.vocab[i] if 0 <= i < len(self.vocab) else "?"
            if parts and re.fullmatch(r"[A-Za-z0-9_]+", word):
                parts.append(" " + word)
            else:
                parts.append(word)
        return "".join(parts)


class FakeTrainableSynthesizer:
    """Stage 2 C double: frozen base logits + trainable LoRA logits.

    ``sft_logits`` is context-free — ``(base + lora)`` expanded over every
    position — so the SFT loss path runs with real autograd that reaches
    only the LoRA parameter, matching the frozen base of the real
    ``HFLoraSynthesizer``. ``answer()`` greedily decodes argmax ids
    (defaulting to an immediate EOS, i.e. an empty C0 answer), wraps the
    result in the Celab wrapper, and goes through the real parser and the
    official HotpotQA token F1. Every call is recorded as
    ``(question, a_report, b_report)`` so tests can assert that C0 and
    Cphi were scored on identical reports; ``answer_fn`` may be set to
    script the output after training.
    """

    def __init__(
        self,
        tokenizer: Optional[FakeChatTokenizer] = None,
        decode: Optional[DecodeConfig] = None,
        base_logits: Optional[Tensor] = None,
        lora_logits: Optional[Tensor] = None,
        answer_fn: Optional[Callable[[str, str, str], str]] = None,
    ):
        self.tokenizer = tokenizer or FakeChatTokenizer()
        self.decode_cfg = decode or DecodeConfig(
            do_sample=False, temperature=0.0, top_p=1.0, max_new_tokens=8
        )
        vocab_size = len(self.tokenizer.vocab)
        if base_logits is None:
            base_logits = torch.zeros(vocab_size)
            base_logits[self.tokenizer.eos_token_id] = 5.0  # C0: empty
        self.base_logits = base_logits.clone().detach()
        if lora_logits is None:
            lora_logits = torch.zeros(vocab_size)
        self.lora_logits = nn.Parameter(lora_logits.clone())
        self.answer_fn = answer_fn
        self.answer_calls: List[Tuple[str, str, str]] = []

    # -- TrainableSynthesizer surface -----------------------------------

    def trainable_parameters(self) -> List[Any]:
        return [self.lora_logits] if self.lora_logits.requires_grad else []

    def sft_logits(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> Tensor:
        combined = (self.base_logits + self.lora_logits).to(input_ids.device)
        return combined[None, None, :].expand(
            input_ids.shape[0], input_ids.shape[1], -1
        ).float()

    def save_adapter(self, adapter_dir: Any) -> None:
        Path(adapter_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            {"lora_logits": self.lora_logits.detach().clone()},
            Path(adapter_dir) / "fake_adapter.pt",
        )

    def load_adapter(self, adapter_dir: Any) -> None:
        path = Path(adapter_dir) / "fake_adapter.pt"
        if not path.is_file():
            raise FileNotFoundError(f"fake adapter not found: {path}")
        state = torch.load(path, map_location="cpu")
        if "lora_logits" not in state:
            raise ValueError("fake adapter state must contain 'lora_logits'")
        with torch.no_grad():
            self.lora_logits.copy_(
                state["lora_logits"].to(
                    self.lora_logits.device, self.lora_logits.dtype
                )
            )

    def state_dict(self) -> Dict[str, Tensor]:
        return {"lora_logits": self.lora_logits.detach().clone()}

    # -- Synthesizer surface ---------------------------------------------

    def answer(
        self,
        messages: List[Dict[str, str]],
        gold: str,
        question: str = "",
        a_report: str = "",
        b_report: str = "",
    ) -> SynthResult:
        self.answer_calls.append((question, a_report, b_report))
        generated: List[int] = []
        if self.answer_fn is not None:
            raw = self.answer_fn(question, a_report, b_report)
            generated = self.tokenizer._tokenize(raw)[0]
        else:
            with torch.no_grad():
                logits = self.base_logits + self.lora_logits.detach()
                for _ in range(self.decode_cfg.max_new_tokens):
                    next_id = int(torch.argmax(logits).item())
                    generated.append(next_id)
                    if next_id == self.tokenizer.eos_token_id:
                        break
            raw = self.tokenizer.decode(generated)
        raw_output = f"{C_WRAPPER_PREFIX}{raw}{C_WRAPPER_SUFFIX}"
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
            input_tokens=0,  # the fake has no real chat template costs
            generated_tokens=len(generated),
            finish_reason="eos",
        )

    def info(self) -> Dict[str, Any]:
        return {
            "synthesizer_kind": "FakeTrainableSynthesizer",
            "vocab_size": len(self.tokenizer.vocab),
            "trainable_parameters": len(self.trainable_parameters()),
        }
