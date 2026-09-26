"""Model engines (spec sec. 12).

``ModelEngine`` is the abstraction the orchestrator talks to; the whole
experiment shares ONE frozen model instance across the three logical agents.

``HFEngine``:
- loads Gemma-3-1B-IT once (fp16 by default), never fine-tunes;
- builds each call's input with the model's own chat template
  (``apply_chat_template``), so input tokens are counted with the REAL
  Gemma tokenizer;
- re-seeds PyTorch with the per-agent, per-call seed supplied by ``Agent``;
- counts generated tokens as ``len(output_ids[prompt_len:])``, which
  includes any EOS token the model emits.

``MockEngine`` is the scripted test double (no GPU): outputs come from a
per-speaker FIFO of scripts, tokens are counted by whitespace-splitting the
script text (the split is reported by ``MockEngine``, so assertions about
token sums are consistent), and every call is recorded for end-to-end
isolation tests. ``MockEngine`` deliberately shares the same
``generate(messages, seed, speaker)`` interface as ``HFEngine``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .seeds import seed_torch

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    speaker: str
    raw_output: str
    input_tokens: int
    generated_tokens: int
    finish_reason: str  # "eos" | "length" (mock always "eos")
    generation_cap_reached: bool
    tokenizer_name: str
    generation_seed: int
    engine: str = "hf"  # "hf" | "mock"


class ModelEngine:
    """Common engine interface: build the input, generate, count tokens."""

    def generate(
        self, messages: List[Dict[str, str]], seed: int, speaker: str
    ) -> GenerationResult:
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        raise NotImplementedError


def _normalize_eos_ids(value: Any) -> List[int]:
    """Normalize a generation-config eos_token_id into a list of ints."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


class HFEngine(ModelEngine):
    """One shared frozen HuggingFace model behind three logical agents."""

    def __init__(
        self,
        model_name: str,
        generation_params: Any,
        dtype: str = "float16",
        device: str = "cuda",
        attn_implementation: Optional[str] = None,
        model_revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        max_input_length: int = 30000,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.dtype_name = dtype
        self.device_name = device
        self.max_input_length = max_input_length
        self.gen_kwargs = generation_params.to_generate_kwargs()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=tokenizer_revision
        )
        self.tokenizer_revision = self.tokenizer.init_kwargs.get(
            "_commit_hash", tokenizer_revision
        )
        load_kwargs: Dict[str, Any] = {}
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        if dtype == "float16":
            load_kwargs["torch_dtype"] = torch.float16
        elif dtype == "bfloat16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif dtype == "float32":
            load_kwargs["torch_dtype"] = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=model_revision, **load_kwargs
        )
        self.model.eval()
        self.model.to(device)

        # Resolved EOS ids: tokenizer EOS plus every entry of the model's
        # generation config (Gemma-3 resolves to something like [1, 106]).
        eos_ids = set()
        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(self.tokenizer.eos_token_id)
        for value in _normalize_eos_ids(
            self.model.generation_config.eos_token_id
        ):
            eos_ids.add(value)
        self.resolved_eos_ids = sorted(eos_ids)

        # Resolved revisions/dtype/attention, recorded verbatim in run records.
        import transformers

        self.model_revision = getattr(
            self.model.config, "_commit_hash", model_revision
        )
        self.resolved_dtype = str(self.model.dtype)
        self.resolved_attn = getattr(
            self.model.config, "_attn_implementation", None
        )
        if self.resolved_attn is None:
            self.resolved_attn = getattr(
                self.model.config, "attn_implementation", "unknown"
            )
        self.transformers_version = transformers.__version__

    def generate(
        self, messages: List[Dict[str, str]], seed: int, speaker: str
    ) -> GenerationResult:
        import torch

        # Agent supplies a stable seed derived from run/speaker/call index.
        seed_torch(seed)

        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_tokens = int(token_ids.shape[1])
        if input_tokens > self.max_input_length:
            raise RuntimeError(
                f"input length {input_tokens} exceeds max_input_length "
                f"{self.max_input_length} (speaker={speaker})"
            )

        input_ids = token_ids.to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                do_sample=self.gen_kwargs["do_sample"],
                temperature=self.gen_kwargs["temperature"],
                top_p=self.gen_kwargs["top_p"],
                max_new_tokens=self.gen_kwargs["max_new_tokens"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.resolved_eos_ids or None,
                use_cache=True,
            )
        generated_ids = outputs[0][input_tokens:].tolist()

        # Token accounting: generated tokens = everything after the prompt,
        # including any EOS token the model emitted (spec sec. 12.4).
        generated_tokens = len(generated_ids)
        if self.resolved_eos_ids:
            ends_with_eos = (
                bool(generated_ids)
                and generated_ids[-1] in self.resolved_eos_ids
            )
        else:
            ends_with_eos = False
        finish_reason = "eos" if ends_with_eos else "length"
        generation_cap_reached = (
            not ends_with_eos
            and generated_tokens >= self.gen_kwargs["max_new_tokens"]
        )
        raw_output = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        return GenerationResult(
            speaker=speaker,
            raw_output=raw_output,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            finish_reason=finish_reason,
            generation_cap_reached=generation_cap_reached,
            tokenizer_name=self.model_name,
            generation_seed=seed,
        )

    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "engine": "transformers.generate",
            "model_class": type(self.model).__name__,
            "model_type": getattr(self.model.config, "model_type", None),
            "num_parameters": int(self.model.num_parameters()),
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "dtype": self.resolved_dtype,
            "attn_implementation": self.resolved_attn,
            "device": self.device_name,
            "resolved_eos_token_ids": self.resolved_eos_ids,
            "tokenizer_eos_token_id": self.tokenizer.eos_token_id,
            "tokenizer_pad_token_id": self.tokenizer.pad_token_id,
            "transformers_version": self.transformers_version,
            "max_input_length": self.max_input_length,
        }


@dataclass
class MockCall:
    speaker: str
    seed: int
    messages: List[Dict[str, str]]
    result: GenerationResult


class MockEngine(ModelEngine):
    """Scripted engine for tests; no model, no GPU.

    ``scripts`` maps a speaker name to a FIFO of raw outputs. Scripts may
    be strings or callables ``f(messages) -> str``. ``cap_speakers`` lists
    speakers whose outputs should be reported as hitting the generation
    cap (finish_reason="length", generation_cap_reached=True). Token counts
    come from whitespace-splitting the script output, so callers can assert
    sums that match the recorded numbers.
    """

    def __init__(
        self,
        scripts: Optional[Dict[str, List[Any]]] = None,
        cap_speakers: Optional[List[str]] = None,
        input_tokens: int = 10,
    ):
        self._queues: Dict[str, List[Any]] = {
            speaker: list(scripts_list)
            for speaker, scripts_list in (scripts or {}).items()
        }
        self.cap_speakers = set(cap_speakers or [])
        self._input_tokens = input_tokens
        self.calls: List[MockCall] = []
        self._calls_by_speaker: Dict[str, List[MockCall]] = {}

    @property
    def calls_by_speaker(self) -> Dict[str, List[MockCall]]:
        return self._calls_by_speaker

    def generate(
        self, messages: List[Dict[str, str]], seed: int, speaker: str
    ) -> GenerationResult:
        queue = self._queues.setdefault(speaker, [])
        if not queue:
            raise RuntimeError(
                f"MockEngine has no scripted output left for speaker {speaker!r}"
            )
        script = queue.pop(0)
        text = script(messages) if callable(script) else script
        capped = speaker in self.cap_speakers
        result = GenerationResult(
            speaker=speaker,
            raw_output=text,
            input_tokens=self._input_tokens,
            generated_tokens=len(text.split()),
            finish_reason="length" if capped else "eos",
            generation_cap_reached=capped,
            tokenizer_name="mock-whitespace-splitter",
            generation_seed=seed,
            engine="mock",
        )
        call = MockCall(speaker=speaker, seed=seed, messages=messages, result=result)
        self.calls.append(call)
        self._calls_by_speaker.setdefault(speaker, []).append(call)
        return result

    def info(self) -> Dict[str, Any]:
        return {
            "engine": "mock",
            "tokenizer_name": "mock-whitespace-splitter",
            "note": "scripted test double; token counts are whitespace splits",
        }
