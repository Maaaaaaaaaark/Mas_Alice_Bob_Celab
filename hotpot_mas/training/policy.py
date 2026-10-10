"""Worker policy abstraction for Stage 1 training.

``Policy`` is the interface the GRPO trainer talks to. Two implementations:

- ``HFPolicy``: a HuggingFace causal LM wrapped with a peft LoRA adapter
  (the workers' shared ``theta``), with a manual token-by-token sampling
  loop that keeps the per-token log probabilities **without** dumping
  full-vocabulary logits. Teacher forcing re-scores a fixed token sequence
  under the *current* policy with gradients enabled.
- ``FakePolicy`` (``fake_policy.py``): a context-free categorical policy
  over a tiny vocabulary, used by the CPU-only unit tests. It implements
  the same interface with real autograd, so the exact GRPO loss code runs
  unchanged offline.

The frozen synthesizer C is *not* a ``Policy`` (it has no trainable
parameters); it lives in ``synthesizers.py``.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
from torch import Tensor

from hotpot_mas.seeds import seed_torch

from .config import DecodeConfig, ModelConfig


@dataclass
class Report:
    """One sampled worker report (old-policy rollout artifact)."""

    token_ids: List[int]
    text: str
    logprobs: List[float]  # old-policy per-token log probs, length T
    num_tokens: int = 0
    finish_reason: str = "eos"  # "eos" | "length"

    def __post_init__(self) -> None:
        if self.num_tokens == 0:
            self.num_tokens = len(self.token_ids)
        if self.num_tokens != len(self.token_ids):
            raise ValueError("num_tokens must match len(token_ids)")
        if len(self.logprobs) != len(self.token_ids):
            raise ValueError("logprobs must match len(token_ids)")


class Policy(Protocol):
    """Minimal policy surface used by rollout and the trainer."""

    def tokenize(self, messages: List[Dict[str, str]]) -> List[int]:
        """Prompt token ids for a message list (generation prompt added)."""
        ...

    def decode(self, token_ids: List[int]) -> str:
        ...

    def sample_report(
        self,
        messages: List[Dict[str, str]],
        generation_seed: int,
        decode: DecodeConfig,
    ) -> Report:
        """Sample one report under the current policy (no grad)."""
        ...

    def decode_report(
        self,
        messages: List[Dict[str, str]],
        generation_seed: int,
        decode: DecodeConfig,
    ) -> Report:
        """Deterministic report for evaluation (``do_sample=False``)."""
        ...

    def teacher_force(
        self,
        input_ids: List[int],
        completion_ids: List[int],
        decode: Optional[DecodeConfig] = None,
    ) -> Tensor:
        """Per-token log probs of ``completion_ids`` under the current
        policy, with gradients enabled. Shape ``[T]``."""
        ...

    def token_log_distributions(
        self,
        input_ids: List[int],
        completion_ids: List[int],
        decode: Optional[DecodeConfig] = None,
    ) -> Tensor:
        """Full-vocabulary log distributions at completion positions."""
        ...

    def state_dict(self) -> Dict[str, Tensor]:
        ...

    def load_state_dict(self, state: Dict[str, Tensor]) -> None:
        ...

    def parameter_norm_summary(self) -> Dict[str, Any]:
        ...

    def info(self) -> Dict[str, Any]:
        ...


def _filter_logits_top_p(logits: Tensor, top_p: float) -> Tensor:
    """Nucleus filtering in place-free style: -inf outside the top-p set."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumsum = sorted_probs.cumsum(dim=-1)
    # Keep the first token that crosses the threshold, drop the rest.
    remove = cumsum > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = sorted_logits.masked_fill(remove, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, sorted_indices, filtered)


def _behavior_log_probs(logits: Tensor, decode: DecodeConfig) -> Tensor:
    """Log probabilities of the exact distribution used for rollout.

    PPO/GRPO ratios are only meaningful when the cached old log probability
    and the re-scored new log probability describe the same distribution.
    Therefore temperature and nucleus filtering are applied here as well as
    in sampling.
    """
    adjusted = logits.float()
    if (
        decode.do_sample
        and decode.temperature > 0.0
        and decode.temperature != 1.0
    ):
        adjusted = adjusted / decode.temperature
    top_p = decode.top_p if decode.do_sample else 1.0
    return torch.log_softmax(_filter_logits_top_p(adjusted, top_p), dim=-1)


class HFPolicy:
    """One base model + one LoRA adapter = the workers' shared policy.

    Loading a checkpoint re-uses the base model and attaches the stored
    adapter, so worker and synthesizer stages never duplicate weights.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
    ):
        import torch as _torch  # noqa: F401  (module loaded once)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_cfg = model_cfg
        self.model_name = model_cfg.name
        self.device_name = model_cfg.device
        self.max_input_length = model_cfg.max_input_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.name, revision=model_cfg.revision
        )
        load_kwargs: Dict[str, Any] = {}
        if model_cfg.attn_implementation:
            load_kwargs["attn_implementation"] = model_cfg.attn_implementation
        if model_cfg.dtype == "float16":
            load_kwargs["torch_dtype"] = torch.float16
        elif model_cfg.dtype == "bfloat16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif model_cfg.dtype == "float32":
            load_kwargs["torch_dtype"] = torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name, revision=model_cfg.revision, **load_kwargs
        )
        base_model.eval()

        from peft import LoraConfig, get_peft_model

        lora_kwargs: Dict[str, Any] = {
            "r": model_cfg.peft.r,
            "lora_alpha": model_cfg.peft.alpha,
            "lora_dropout": model_cfg.peft.dropout,
            "bias": "none",
        }
        lora_kwargs["target_modules"] = (
            model_cfg.peft.target_modules
            if model_cfg.peft.target_modules is not None
            else "all-linear"
        )
        self.model = get_peft_model(
            base_model, LoraConfig(task_type="CAUSAL_LM", **lora_kwargs)
        )
        self.model.to(model_cfg.device)
        base_forward = self.model.get_base_model().forward
        self.supports_logits_to_keep = "logits_to_keep" in inspect.signature(
            base_forward
        ).parameters

        chat_template = getattr(self.tokenizer, "chat_template", None)
        if not chat_template:
            raise RuntimeError(
                f"tokenizer for {model_cfg.name!r} has no chat template; "
                "refusing to inject a custom one"
            )
        self.chat_template_sha256 = hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest()

        # Resolved EOS ids: tokenizer EOS plus the model generation config.
        eos_ids = set()
        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(self.tokenizer.eos_token_id)
        gen_cfg = getattr(self.model, "generation_config", None)
        gen_eos = getattr(gen_cfg, "eos_token_id", None)
        if gen_eos is not None:
            if isinstance(gen_eos, int):
                eos_ids.add(gen_eos)
            else:
                eos_ids.update(int(v) for v in gen_eos)
        self.resolved_eos_ids = sorted(eos_ids)

    # -- tokenization ---------------------------------------------------

    def tokenize(self, messages: List[Dict[str, str]]) -> List[int]:
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return [int(i) for i in ids]

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=True
        )

    def _check_input_length(self, prompt_len: int, speaker: str) -> None:
        if prompt_len > self.max_input_length:
            raise RuntimeError(
                f"input length {prompt_len} exceeds max_input_length "
                f"{self.max_input_length} (speaker={speaker})"
            )

    # -- sampling -------------------------------------------------------

    def _sample_loop(
        self,
        prompt_ids: List[int],
        generation_seed: int,
        decode: DecodeConfig,
    ) -> Report:
        self._check_input_length(len(prompt_ids), "worker")
        seed_torch(generation_seed)
        input_ids = torch.tensor([prompt_ids], device=self.device_name)
        past_key_values = None
        generated: List[int] = []
        logprobs: List[float] = []
        finish_reason = "length"

        with torch.no_grad():
            for _ in range(decode.max_new_tokens):
                step_input = input_ids if past_key_values is None else input_ids[:, -1:]
                outputs = self.model(
                    input_ids=step_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[0, -1].float()
                log_probs = _behavior_log_probs(logits, decode)
                if decode.do_sample:
                    next_id = int(torch.multinomial(log_probs.exp(), 1).item())
                else:
                    next_id = int(torch.argmax(log_probs).item())
                logprobs.append(float(log_probs[next_id].item()))
                generated.append(next_id)
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_id]], device=self.device_name)],
                    dim=1,
                )
                if next_id in self.resolved_eos_ids:
                    finish_reason = "eos"
                    break

        return Report(
            token_ids=generated,
            text=self.decode(generated),
            logprobs=logprobs,
            finish_reason=finish_reason,
        )

    def sample_report(
        self,
        messages: List[Dict[str, str]],
        generation_seed: int,
        decode: DecodeConfig,
    ) -> Report:
        return self._sample_loop(
            self.tokenize(messages), generation_seed, decode
        )

    def decode_report(
        self,
        messages: List[Dict[str, str]],
        generation_seed: int,
        decode: DecodeConfig,
    ) -> Report:
        # Evaluation decoding is deterministic by configuration; the same
        # loop with do_sample=False returns logprobs as a diagnostic bonus.
        return self._sample_loop(
            self.tokenize(messages), generation_seed, decode
        )

    # -- teacher forcing ------------------------------------------------

    def _completion_logits(
        self, input_ids: List[int], completion_ids: List[int]
    ) -> Tensor:
        """Return logits predicting exactly the completion tokens.

        Newer Transformers models accept ``logits_to_keep`` and avoid
        materializing prompt-position vocabulary logits. The fallback keeps
        compatibility with older model implementations.
        """
        if not completion_ids:
            raise ValueError("completion_ids must be non-empty")
        if not input_ids:
            raise ValueError("input_ids must be non-empty for a causal LM")
        full = torch.tensor(
            [input_ids + list(completion_ids)], device=self.device_name
        )
        kwargs: Dict[str, Any] = {"input_ids": full}
        supports_logits_to_keep = getattr(
            self, "supports_logits_to_keep", False
        )
        if supports_logits_to_keep:
            # Causal logits at position p predict token p+1. Keep one extra
            # position so the final prompt position can score completion[0],
            # then discard the logits at the final completion position.
            kwargs["logits_to_keep"] = len(completion_ids) + 1
        outputs = self.model(**kwargs)
        logits = outputs.logits[0].float()
        if supports_logits_to_keep:
            if logits.shape[0] != len(completion_ids) + 1:
                raise RuntimeError(
                    "model logits_to_keep returned an unexpected sequence length"
                )
            return logits[:-1]
        start = len(input_ids)
        positions = torch.arange(
            start - 1,
            start + len(completion_ids) - 1,
            device=self.device_name,
        )
        return logits[positions]

    def teacher_force(
        self,
        input_ids: List[int],
        completion_ids: List[int],
        decode: Optional[DecodeConfig] = None,
    ) -> Tensor:
        """Log probs of each completion token under the current policy.

        Returns shape ``[T]`` with gradients flowing into the LoRA
        parameters only (the base model is frozen).
        """
        decode = decode or DecodeConfig()
        logits = self._completion_logits(input_ids, completion_ids)
        behavior_log_probs = _behavior_log_probs(logits, decode)
        return behavior_log_probs[
            torch.arange(len(completion_ids), device=self.device_name),
            torch.tensor(completion_ids, device=self.device_name),
        ]

    def token_log_distributions(
        self,
        input_ids: List[int],
        completion_ids: List[int],
        decode: Optional[DecodeConfig] = None,
    ) -> Tensor:
        """Full categorical behavior-policy log probabilities, no gradient."""
        self.model.eval()
        with torch.inference_mode():
            logits = self._completion_logits(input_ids, completion_ids)
            return _behavior_log_probs(
                logits, decode or DecodeConfig()
            ).detach()

    # -- checkpointing --------------------------------------------------

    def save_adapter(self, adapter_dir: Path) -> None:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(adapter_dir)

    def load_adapter(self, adapter_dir: Path) -> None:
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"adapter directory not found: {adapter_dir}")
        # ``get_peft_model`` already created the active ``default`` adapter.
        # Load weights into it in place; ``PeftModel.load_adapter`` creates a
        # new named adapter and requires an adapter name in current PEFT.
        from peft.utils.save_and_load import (
            load_peft_weights,
            set_peft_model_state_dict,
        )

        state = load_peft_weights(str(adapter_dir), device=self.device_name)
        result = set_peft_model_state_dict(
            self.model, state, adapter_name="default"
        )
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if unexpected:
            raise RuntimeError(
                "unexpected keys while loading worker adapter: "
                + ", ".join(unexpected[:10])
            )

    # -- parameter bookkeeping ------------------------------------------

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def state_dict(self) -> Dict[str, Tensor]:
        return {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def load_state_dict(self, state: Dict[str, Tensor]) -> None:
        trainable = {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        if set(state) != set(trainable):
            missing = set(trainable) - set(state)
            extra = set(state) - set(trainable)
            raise ValueError(
                f"state_dict key mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        with torch.no_grad():
            for name, param in trainable.items():
                param.copy_(state[name].to(param.device, param.dtype))

    def parameter_norm_summary(self) -> Dict[str, Any]:
        total = 0.0
        count = 0
        per_module: Dict[str, float] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            l2 = float(param.detach().float().norm().item())
            per_module[name] = l2
            total += l2**2
            count += param.numel()
        return {
            "trainable_parameters": len(per_module),
            "trainable_numel": count,
            "total_l2_norm": total**0.5,
            "per_parameter_l2": per_module,
        }

    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "policy_kind": "HFPolicy",
            "peft_r": self.model_cfg.peft.r,
            "peft_alpha": self.model_cfg.peft.alpha,
            "device": self.device_name,
            "dtype": self.model_cfg.dtype,
            "max_input_length": self.max_input_length,
            "chat_template_sha256": self.chat_template_sha256,
            "resolved_eos_ids": self.resolved_eos_ids,
            "trainable_parameters": len(self.trainable_parameters()),
        }


def remap_optimizer_state(
    optimizer: torch.optim.Optimizer, param_names: List[str]
) -> Dict[str, Any]:
    """Serialize an optimizer state for a later rebuild on new tensors.

    Stores one entry per trainable parameter (by name, in order) plus the
    parameter-group hyperparameters. The trainer rebuilds the optimizer on
    the freshly loaded policy and maps state by position.
    """
    states = [optimizer.state[p] for p in optimizer.param_groups[0]["params"]]
    serializable = [
        {k: v.detach().cpu() if isinstance(v, Tensor) else v
         for k, v in state.items()}
        for state in states
    ]
    return {
        "param_names": list(param_names),
        "optimizer_states": serializable,
        "param_groups": [
            {k: v for k, v in group.items() if k != "params"}
            for group in optimizer.param_groups
        ],
    }


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    saved: Dict[str, Any],
) -> None:
    """Load optimizer state saved by :func:`remap_optimizer_state`."""
    params = list(optimizer.param_groups[0]["params"])
    if len(params) != len(saved["optimizer_states"]):
        raise ValueError(
            "optimizer parameter count mismatch on restore "
            f"({len(params)} vs {len(saved['optimizer_states'])})"
        )
    for param, state in zip(params, saved["optimizer_states"]):
        optimizer.state[param] = {
            k: v.to(param.device) if isinstance(v, Tensor) else v
            for k, v in state.items()
        }
    for group, saved_group in zip(
        optimizer.param_groups, saved["param_groups"]
    ):
        for key, value in saved_group.items():
            group[key] = value
