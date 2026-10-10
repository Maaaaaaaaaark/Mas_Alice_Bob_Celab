"""The synthesizer C: frozen base model, greedy decoding, deterministic
reward.

C has **no** trainable parameters during Stage 1 (it is the reward
evaluator) and gets its own independent LoRA only in Stage 2. It takes the
question plus one report from each worker and outputs a single answer,
which is scored against the gold answer with the official HotpotQA token
F1.

The gold answer never enters any C input; it appears only here, as the
reward/evaluation target.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
from torch import Tensor

from hotpot_mas.evaluation import token_overlap_scores
from hotpot_mas.parser import extract_final_answer

from .config import DecodeConfig, ModelConfig


@dataclass
class SynthResult:
    """One C answer plus its reward components."""

    question: str
    a_report: str
    b_report: str
    gold: str
    raw_output: str
    pred_answer: str  # parsed <FINAL> body, or raw output fallback
    parsed: bool  # True when the <FINAL> tag parsed successfully
    precision: float
    recall: float
    f1: float
    input_tokens: int
    generated_tokens: int
    finish_reason: str  # "eos" | "length"


class Synthesizer(Protocol):
    def answer(
        self,
        messages: List[Dict[str, str]],
        gold: str,
        question: str = "",
        a_report: str = "",
        b_report: str = "",
    ) -> SynthResult:
        ...

    def info(self) -> Dict[str, Any]:
        ...


class HFSynthesizer:
    """Frozen HF base model (no LoRA) decoding greedily."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        decode: DecodeConfig,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_cfg.name
        self.device_name = model_cfg.device
        self.decode_cfg = decode
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name, revision=model_cfg.revision, **load_kwargs
        )
        self.model.eval()
        self.model.to(model_cfg.device)
        self.supports_logits_to_keep = "logits_to_keep" in inspect.signature(
            self.model.forward
        ).parameters

        # Resolved EOS ids, same rule as HFEngine.
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

        chat_template = getattr(self.tokenizer, "chat_template", None)
        if not chat_template:
            raise RuntimeError(
                f"tokenizer for {model_cfg.name!r} has no chat template"
            )
        self.chat_template_sha256 = hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest()

    def tokenize(self, messages: List[Dict[str, str]]) -> List[int]:
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return [int(i) for i in ids]

    def _prefix_ids(self) -> List[int]:
        if not self.decode_cfg.answer_prefix:
            return []
        return [
            int(value)
            for value in self.tokenizer.encode(
                self.decode_cfg.answer_prefix,
                add_special_tokens=False,
            )
        ]

    @staticmethod
    def _strip_answer_labels(text: str) -> str:
        """Remove protocol labels without changing answer-internal text."""
        cleaned = text.strip().splitlines()[0].strip() if text.strip() else ""
        label = re.compile(r"^(?:answer|celab|reader)\s*:\s*", re.IGNORECASE)
        while label.match(cleaned):
            cleaned = label.sub("", cleaned, count=1).strip()
        final = extract_final_answer(cleaned)
        if final.action == "final" and final.body is not None:
            cleaned = final.body.strip()
        return cleaned

    def _decode_constrained(
        self, prompt_ids: List[int]
    ) -> Tuple[str, List[int], str, int]:
        """Greedy continuation after a fixed prefix, optionally to newline."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        prefix_ids = self._prefix_ids()
        model_input = prompt_ids + prefix_ids
        input_ids = torch.tensor([model_input], device=self.device_name)
        stop_state: Dict[str, bool] = {"newline": False}

        class StopAtNewline(StoppingCriteria):
            def __call__(inner_self, generated_ids, scores, **kwargs):
                continuation = generated_ids[0, len(model_input):].tolist()
                text = self.tokenizer.decode(
                    continuation, skip_special_tokens=True
                )
                found = "\n" in text or "\r" in text
                stop_state["newline"] = found
                return found

        stopping = None
        if self.decode_cfg.stop_on_newline:
            stopping = StoppingCriteriaList([StopAtNewline()])
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=self.decode_cfg.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.resolved_eos_ids or None,
                use_cache=True,
                stopping_criteria=stopping,
            )
        generated = outputs[0][len(model_input):].tolist()
        continuation = self.tokenizer.decode(
            generated, skip_special_tokens=True
        )
        if self.decode_cfg.stop_on_newline:
            continuation = re.split(r"[\r\n]", continuation, maxsplit=1)[0]
        raw_output = self.decode_cfg.answer_prefix + continuation
        if stop_state["newline"]:
            finish_reason = "newline"
        elif generated and generated[-1] in self.resolved_eos_ids:
            finish_reason = "eos"
        else:
            finish_reason = "length"
        return raw_output, generated, finish_reason, len(model_input)

    def answer(
        self,
        messages: List[Dict[str, str]],
        gold: str,
        question: str = "",
        a_report: str = "",
        b_report: str = "",
    ) -> SynthResult:
        import torch

        # SFT forwards put the trainable C in training mode.  Generation
        # must explicitly return to eval mode, especially for future base
        # models that contain dropout.
        self.model.eval()
        prompt_ids = self.tokenize(messages)
        full_prompt_length = len(prompt_ids) + len(self._prefix_ids())
        if full_prompt_length > self.max_input_length:
            raise RuntimeError(
                f"C input length {full_prompt_length} exceeds "
                f"max_input_length {self.max_input_length}"
            )
        raw_output, generated, finish_reason, input_length = (
            self._decode_constrained(prompt_ids)
        )
        if self.decode_cfg.strip_answer_labels:
            pred_answer = self._strip_answer_labels(raw_output)
            parsed_ok = bool(pred_answer)
        else:
            parsed = extract_final_answer(raw_output)
            if parsed.action == "final" and parsed.body is not None:
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
            input_tokens=input_length,
            generated_tokens=len(generated),
            finish_reason=finish_reason,
        )

    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "synthesizer_kind": "HFSynthesizer",
            "frozen": True,
            "decode": "greedy (do_sample=False)",
            "max_new_tokens": self.decode_cfg.max_new_tokens,
            "answer_prefix": self.decode_cfg.answer_prefix,
            "stop_on_newline": self.decode_cfg.stop_on_newline,
            "strip_answer_labels": self.decode_cfg.strip_answer_labels,
            "device": self.device_name,
            "max_input_length": self.max_input_length,
            "chat_template_sha256": self.chat_template_sha256,
            "resolved_eos_ids": self.resolved_eos_ids,
        }

    def gold_answer_log_likelihood(
        self,
        messages: List[Dict[str, str]],
        gold: str,
    ) -> Dict[str, Any]:
        """Teacher-forced gold likelihood after the configured prefix."""
        self.model.eval()
        prompt_ids = self.tokenize(messages) + self._prefix_ids()
        answer_text = (" " if self.decode_cfg.answer_prefix else "") + gold.strip()
        answer_ids = [
            int(value)
            for value in self.tokenizer.encode(
                answer_text, add_special_tokens=False
            )
        ]
        if not answer_ids:
            raise ValueError("gold answer tokenized to an empty sequence")
        if len(prompt_ids) + len(answer_ids) > self.max_input_length:
            raise RuntimeError("gold-likelihood input exceeds max_input_length")
        full = torch.tensor(
            [prompt_ids + answer_ids], device=self.device_name
        )
        with torch.inference_mode():
            if self.supports_logits_to_keep:
                logits = self.model(
                    input_ids=full,
                    logits_to_keep=len(answer_ids) + 1,
                ).logits[0].float()[:-1]
            else:
                all_logits = self.model(input_ids=full).logits[0].float()
                start = len(prompt_ids)
                positions = torch.arange(
                    start - 1,
                    start + len(answer_ids) - 1,
                    device=self.device_name,
                )
                logits = all_logits[positions]
            log_probs = torch.log_softmax(logits, dim=-1)
            targets = torch.tensor(answer_ids, device=self.device_name)
            token_log_probs = log_probs[
                torch.arange(len(answer_ids), device=self.device_name), targets
            ]
        return {
            "token_log_probs": token_log_probs.detach().cpu().tolist(),
            "num_answer_tokens": len(answer_ids),
            "sum_log_likelihood": float(token_log_probs.sum().item()),
            "mean_log_likelihood": float(token_log_probs.mean().item()),
        }


class TrainableSynthesizer(Protocol):
    """Synthesizer surface used by the Stage 2 SFT trainer.

    Everything the trainer needs beyond the evaluation ``Synthesizer``
    interface: a tokenizer for building D_C, logits for the SFT loss, and
    adapter persistence for checkpoints.
    """

    tokenizer: Any  # HF tokenizer surface (apply_chat_template + __call__)

    def trainable_parameters(self) -> List[Any]:
        """Parameters receiving gradients (C's LoRA ``phi`` only)."""
        ...

    def sft_logits(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> Tensor:
        """Full-vocabulary logits ``[B, T, V]`` with gradients enabled."""
        ...

    def save_adapter(self, adapter_dir: Path) -> None:
        ...

    def load_adapter(self, adapter_dir: Path) -> None:
        ...


class HFLoraSynthesizer(HFSynthesizer):
    """Stage 2 C: the frozen base model plus C's own trainable LoRA ``phi``.

    A freshly constructed instance has all-zero ``lora_B`` (verified in
    ``_assert_fresh_lora``), so with dropout disabled its outputs are
    numerically identical to the base model — the step-0 evaluation of this
    instance *is* C0, no second copy of the weights needed.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        decode: DecodeConfig,
    ):
        super().__init__(model_cfg, decode)
        from peft import LoraConfig, get_peft_model

        self.peft_cfg = model_cfg.peft

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
            self.model, LoraConfig(task_type="CAUSAL_LM", **lora_kwargs)
        )
        self.model.eval()
        self.model.to(model_cfg.device)
        self._assert_fresh_lora()

    def _assert_fresh_lora(self) -> None:
        """Refuse to start SFT from anything but a fresh zero-init LoRA.

        The step-0 baseline (C0) is defined as the base model; peft zero-
        initializes every ``lora_B``, which makes a fresh adapter exactly
        the base model. A non-zero ``lora_B`` here means an already-trained
        adapter was injected, silently corrupting C0.
        """
        bad = [
            name
            for name, param in self.model.named_parameters()
            if ("lora_B" in name or "lora_embedding_B" in name)
            and torch.count_nonzero(param.detach()) != 0
        ]
        if bad:
            raise RuntimeError(
                "HFLoraSynthesizer built with a non-fresh LoRA: nonzero "
                f"lora_B parameters {bad[:5]}; C0 would not equal the base "
                "model, refusing to continue"
            )

    def trainable_parameters(self) -> List[Any]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def sft_logits(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> Tensor:
        """Full-vocabulary logits for the SFT loss, gradients enabled."""
        self.model.train()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits.float()

    def save_adapter(self, adapter_dir: Path) -> None:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(adapter_dir)

    def load_adapter(self, adapter_dir: Path) -> None:
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"adapter directory not found: {adapter_dir}")
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
                "unexpected keys while loading synthesizer adapter: "
                + ", ".join(unexpected[:10])
            )

    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "synthesizer_kind": "HFLoraSynthesizer",
            "frozen_base": True,
            "trainable": "lora only",
            "peft_r": self.peft_cfg.r,
            "peft_alpha": self.peft_cfg.alpha,
            "peft_dropout": self.peft_cfg.dropout,
            "decode": "greedy (do_sample=False)",
            "max_new_tokens": self.decode_cfg.max_new_tokens,
            "device": self.device_name,
            "max_input_length": self.max_input_length,
            "chat_template_sha256": self.chat_template_sha256,
            "resolved_eos_ids": self.resolved_eos_ids,
            "trainable_parameters": len(self.trainable_parameters()),
        }
