"""HF model wrapper for Stage 1a worker SFT.

Loads the base Gemma-3-1B-IT model and applies a FRESH LoRA adapter on
top. No old GRPO or synthesizer adapter is ever loaded here — the only
adapter load path is ``load_adapter``, used exclusively to resume THIS
run's own checkpoint. The base weights are verified frozen (requires_grad
audit plus a parameter hash before/after training).

Precision: ``bfloat16`` when the device supports it; otherwise
``float16`` (GPU) or ``float32`` (CPU). The fallback and its reason are
recorded in the run info, never silently assumed.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from .config import WorkerSFTConfig


class HFLoraWorkerModel:
    """Base model + fresh LoRA adapter + tokenizer."""

    def __init__(self, cfg: WorkerSFTConfig) -> None:
        import torch as _torch  # noqa: F401 - device checks below
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.device_name = cfg.model.device
        self._dtype, dtype_reason = self._resolve_dtype()
        self.dtype_reason = dtype_reason

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            revision=cfg.model.revision,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self._dtype]
        self.base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            revision=cfg.model.revision,
            torch_dtype=torch_dtype,
        )

        # Fresh adapter only: nothing is loaded from any previous
        # experiment.
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=cfg.model.peft.r,
            lora_alpha=cfg.model.peft.alpha,
            lora_dropout=cfg.model.peft.dropout,
            target_modules=cfg.model.peft.target_modules or "all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.to(self.device_name)

        self._resolved_lora_modules = self._resolve_lora_modules()
        if not self._resolved_lora_modules:
            raise ValueError(
                "no LoRA modules were attached: verify target_modules "
                "against the actual model"
            )
        self._frozen_audit = self._audit_frozen_base()
        if not self._frozen_audit["base_frozen"]:
            raise ValueError(
                "base model parameters are not frozen: "
                f"{self._frozen_audit['unfrozen_base']}"
            )
        self.chat_template_sha256 = hashlib.sha256(
            (self.tokenizer.chat_template or "").encode("utf-8")
        ).hexdigest()
        self.model_name = cfg.model.name

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _resolve_dtype(self) -> tuple:
        requested = self.cfg.model.dtype
        if self.device_name == "cpu":
            return "float32", "CPU device: float32"
        if requested == "bfloat16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return "bfloat16", "bfloat16 supported by the GPU"
            if torch.cuda.is_available():
                return "float16", (
                    "bfloat16 not supported by this GPU; fell back to "
                    "float16"
                )
        return requested, f"configured dtype {requested}"

    def _resolve_lora_modules(self) -> List[str]:
        modules = []
        for name, _ in self.model.named_modules():
            if name.endswith("lora_A") or name.endswith("lora_B"):
                modules.append(name)
        return sorted(modules)

    def _audit_frozen_base(self) -> Dict[str, Any]:
        unfrozen = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and "lora" not in name:
                unfrozen.append(name)
        trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
        return {
            "base_frozen": not unfrozen,
            "unfrozen_base": unfrozen,
            "trainable_names": sorted(trainable),
        }

    # ------------------------------------------------------------------
    # Model protocol used by the trainer
    # ------------------------------------------------------------------

    def logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Logits [B, T, V] for teacher forcing (autograd intact)."""
        output = self.model(
            input_ids=input_ids.to(self.device_name),
            attention_mask=attention_mask.to(self.device_name),
        )
        return output.logits

    def trainable_parameters(self) -> List[Tensor]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def base_parameters(self) -> List[Tensor]:
        return [p for p in self.model.parameters() if not p.requires_grad]

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

    def trainable_snapshot(self) -> Dict[str, Tensor]:
        """Detached copies of trainable params (for delta norms)."""
        return {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def trainable_delta_norm(
        self, snapshot: Dict[str, Tensor]
    ) -> float:
        total = 0.0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            delta = (param.detach() - snapshot[name].to(param.device)).float()
            total += float(delta.norm().item()) ** 2
        return total**0.5

    def base_param_hash(self, sample_every: Optional[int] = None) -> str:
        """Hash frozen base parameters (all of them, or every k-th).

        Identical hashes before and after training are evidence the base
        weights did not change. ``sample_every`` trades completeness for
        speed in large runs; the trace mode uses the full hash.
        """
        hasher = hashlib.sha256()
        count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                continue
            if sample_every is not None and count % sample_every != 0:
                count += 1
                continue
            hasher.update(name.encode("utf-8"))
            hasher.update(
                param.detach().cpu().flatten().view(torch.uint8).numpy().tobytes()
            )
            count += 1
        return hasher.hexdigest()

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Deterministic greedy generation with finish-reason reporting.

        Manual argmax loop; stops at the end-of-turn token or after
        ``max_new_tokens``. No stop-on-newline: valid targets can have
        two lines.
        """
        if seed is not None:
            import torch as _t

            _t.manual_seed(seed)
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device_name)
        prompt_len = ids.shape[1]
        eos = self.tokenizer.eos_token_id
        generated: List[int] = []
        with torch.no_grad():
            past = None
            current = ids
            for _ in range(max_new_tokens):
                output = self.model(input_ids=current, use_cache=True)
                past = output.past_key_values
                next_logits = output.logits[:, -1, :]
                next_id = int(next_logits.argmax(dim=-1).item())
                generated.append(next_id)
                if eos is not None and next_id == int(eos):
                    break
                current = torch.tensor(
                    [[next_id]], device=self.device_name, dtype=torch.long
                )
        finish_reason = (
            "eos" if generated and generated[-1] == int(eos) else "length"
        )
        token_ids = ids.tolist()[0] + generated
        text = self.tokenizer.decode(
            token_ids[prompt_len:], skip_special_tokens=True
        )
        return {
            "text": text,
            "token_ids": token_ids,
            "new_token_ids": generated,
            "num_new_tokens": len(generated),
            "finish_reason": finish_reason,
        }

    def save_adapter(self, directory: Any) -> None:
        self.model.save_pretrained(str(directory))

    def load_adapter(self, directory: Any) -> None:
        """Load ONLY this run's own checkpoint adapter (resume)."""
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(
            self.base_model, str(directory), is_trainable=True
        )
        self.model.to(self.device_name)

    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device_name,
            "dtype": self._dtype,
            "dtype_reason": self.dtype_reason,
            "peft_r": self.cfg.model.peft.r,
            "peft_alpha": self.cfg.model.peft.alpha,
            "peft_dropout": self.cfg.model.peft.dropout,
            "peft_target_modules": self.cfg.model.peft.target_modules,
            "resolved_lora_modules": self._resolved_lora_modules,
            "frozen_audit": self._frozen_audit,
            "chat_template_sha256": self.chat_template_sha256,
            "max_input_tokens": self.cfg.model.max_input_tokens,
            "fresh_adapter": True,
            "loaded_previous_adapter": False,
        }
