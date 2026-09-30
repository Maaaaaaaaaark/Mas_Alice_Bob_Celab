"""Model engines (spec sec. 12).

``ModelEngine`` is the abstraction the orchestrator talks to. Legacy
experiments share one frozen model instance, while independent-instance
experiments construct one engine per agent on explicitly configured devices.

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

import hashlib
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
    engine: str = "hf"  # "hf" | "vllm" | "mock"


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
    """One frozen HuggingFace model instance on one configured device."""

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


def _vllm_generated_token_count(
    token_ids: List[int],
    finish_reason: Optional[str],
    stop_reason: Any,
    eos_token_ids: List[int],
) -> int:
    """Count vLLM output tokens using the HF experiment's EOS-inclusive rule.

    vLLM versions can omit a special EOS token from ``CompletionOutput`` even
    though it was sampled.  Its API documents ``finish_reason='stop'`` with a
    null ``stop_reason`` as EOS termination.  Add one only when EOS caused the
    stop and the returned IDs do not already contain an EOS token.
    """
    count = len(token_ids)
    eos_stop = finish_reason == "stop" and (
        stop_reason is None or stop_reason in eos_token_ids
    )
    returned_eos = bool(token_ids) and token_ids[-1] in eos_token_ids
    if eos_stop and not returned_eos:
        count += 1
    return count


class VLLMEngine(ModelEngine):
    """Offline vLLM backend using the checkpoint's own chat template.

    This is a separate experimental backend, not a silent replacement for
    ``HFEngine``.  Model/tokenizer revisions and the exact chat-template hash
    are recorded so its runs cannot be mixed with Transformers runs.
    """

    def __init__(
        self,
        model_name: str,
        generation_params: Any,
        dtype: str = "float16",
        model_revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        max_input_length: int = 30000,
        gpu_memory_utilization: float = 0.80,
        enforce_eager: bool = False,
    ):
        try:
            import vllm
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "engine='vllm' requires the optional vLLM dependency; "
                "install requirements-vllm.txt in a separate environment"
            ) from exc
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig

        self.model_name = model_name
        self.gen_kwargs = generation_params.to_generate_kwargs()
        self.dtype_name = dtype
        self.max_input_length = max_input_length
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.vllm_version = vllm.__version__

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=tokenizer_revision
        )
        self.hf_config = AutoConfig.from_pretrained(
            model_name, revision=model_revision
        )
        try:
            generation_config = GenerationConfig.from_pretrained(
                model_name, revision=model_revision
            )
        except OSError:
            generation_config = None

        chat_template = getattr(self.tokenizer, "chat_template", None)
        if not chat_template:
            raise RuntimeError(
                f"tokenizer for {model_name!r} has no chat template; "
                "refusing to guess or inject a custom template"
            )
        self.chat_template_sha256 = hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest()

        eos_ids = set(_normalize_eos_ids(self.tokenizer.eos_token_id))
        if generation_config is not None:
            eos_ids.update(
                _normalize_eos_ids(generation_config.eos_token_id)
            )
        self.resolved_eos_ids = sorted(eos_ids)
        self.model_revision = getattr(
            self.hf_config, "_commit_hash", model_revision
        )
        self.tokenizer_revision = self.tokenizer.init_kwargs.get(
            "_commit_hash", tokenizer_revision
        )

        self.llm = LLM(
            model=model_name,
            tokenizer=model_name,
            revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            dtype=dtype,
            tensor_parallel_size=1,
            max_model_len=max_input_length + self.gen_kwargs["max_new_tokens"],
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            trust_remote_code=False,
            seed=0,
        )

    def generate(
        self, messages: List[Dict[str, str]], seed: int, speaker: str
    ) -> GenerationResult:
        from vllm import SamplingParams

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        input_tokens = len(prompt_ids)
        if input_tokens > self.max_input_length:
            raise RuntimeError(
                f"input length {input_tokens} exceeds max_input_length "
                f"{self.max_input_length} (speaker={speaker})"
            )

        temperature = (
            self.gen_kwargs["temperature"]
            if self.gen_kwargs["do_sample"]
            else 0.0
        )
        sampling = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=self.gen_kwargs["top_p"],
            max_tokens=self.gen_kwargs["max_new_tokens"],
            seed=seed,
            skip_special_tokens=True,
        )
        requests = self.llm.chat(
            messages,
            sampling_params=sampling,
            use_tqdm=False,
            add_generation_prompt=True,
        )
        if len(requests) != 1 or len(requests[0].outputs) != 1:
            raise RuntimeError("vLLM returned an unexpected number of outputs")
        request = requests[0]
        output = request.outputs[0]
        actual_prompt_ids = request.prompt_token_ids
        if actual_prompt_ids is None:
            raise RuntimeError("vLLM did not return prompt_token_ids")
        if list(actual_prompt_ids) != list(prompt_ids):
            raise RuntimeError(
                "vLLM and Transformers rendered different official chat-template "
                "token IDs; refusing an uncontrolled comparison"
            )

        token_ids = list(output.token_ids)
        generated_tokens = _vllm_generated_token_count(
            token_ids,
            output.finish_reason,
            output.stop_reason,
            self.resolved_eos_ids,
        )
        if output.finish_reason == "stop":
            finish_reason = "eos"
        elif output.finish_reason == "length":
            finish_reason = "length"
        else:
            raise RuntimeError(
                f"unexpected vLLM finish_reason: {output.finish_reason!r}"
            )
        generation_cap_reached = (
            finish_reason == "length"
            and generated_tokens >= self.gen_kwargs["max_new_tokens"]
        )
        return GenerationResult(
            speaker=speaker,
            raw_output=output.text,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            finish_reason=finish_reason,
            generation_cap_reached=generation_cap_reached,
            tokenizer_name=self.model_name,
            generation_seed=seed,
            engine="vllm",
        )

    def info(self) -> Dict[str, Any]:
        architectures = getattr(self.hf_config, "architectures", None) or []
        return {
            "model_name": self.model_name,
            "engine": "vllm.LLM.chat",
            "model_class": architectures[0] if architectures else None,
            "model_type": getattr(self.hf_config, "model_type", None),
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "dtype": self.dtype_name,
            "device": "cuda",
            "tensor_parallel_size": 1,
            "vllm_version": self.vllm_version,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "chat_template_source": "pinned Hugging Face tokenizer_config",
            "chat_template_sha256": self.chat_template_sha256,
            "resolved_eos_token_ids": self.resolved_eos_ids,
            "token_count_policy": (
                "len(vllm CompletionOutput.token_ids), plus one only when "
                "EOS caused stop but EOS is omitted from returned IDs"
            ),
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
