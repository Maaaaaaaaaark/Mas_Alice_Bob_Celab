"""CPU-only fake tokenizer/model for Stage 1a tests.

These stand in for the HuggingFace model and tokenizer so the conversion,
masking, loss, accumulation, and trace code can be exercised on a machine
with no GPU and no 1B-parameter checkpoint. They implement the SAME
protocol the trainer uses (``.model`` with eval/train/named_parameters,
``logits()`` with autograd, ``generate()``, ``save_adapter``/
``load_adapter``, ``info()``) and the SAME chat-template contract the
masking code verifies (the full conversation extends the generation
prompt at string AND token level, and the assistant turn ends with the
end-of-turn token).

A fake-model run is a functional test ONLY. It is not a training result;
the trainer records ``real_model=False`` for it, and the worked example
states the flag explicitly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn

SPECIAL_TOKENS = [
    "<bos>",
    "<eos>",
    "<pad>",
    "<unk>",
    "<assistant>",
    "<assistant_end>",
]

EOS_STR = "<eos>"


class FakeWorkerTokenizer:
    """Whitespace tokenizer with a chat template matching the real one.

    Non-assistant messages render as ``role: content`` lines; the
    generation prompt adds a trailing ``assistant: ``; an assistant
    message appends ``assistant: content <eos>`` so the full conversation
    extends the prompt exactly and ends with the end-of-turn token.
    """

    def __init__(self, corpus: Optional[List[str]] = None) -> None:
        self._corpus = list(corpus or [])
        self.vocab: Dict[str, int] = {}
        for token in SPECIAL_TOKENS:
            self.vocab[token] = len(self.vocab)
        # Deterministic ids: first appearance order in the corpus.
        for text in self._corpus:
            for token in self._tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        self._ids: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.eos_token_id = self.vocab["<eos>"]
        self.pad_token_id = self.vocab["<pad>"]
        self.unk_token_id = self.vocab["<unk>"]
        self.bos_token_id = self.vocab["<bos>"]
        self.chat_template = (
            "fake template: non-assistant messages as 'role: content' "
            "lines; assistant turn 'assistant: content <eos>'; generation "
            "prompt ends with 'assistant: '"
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.split()

    def _id_of(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def _render(
        self, messages: List[Dict[str, str]], add_generation_prompt: bool
    ) -> str:
        lines: List[str] = []
        has_assistant = False
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "assistant":
                lines.append(f"assistant: {content} {EOS_STR}")
                has_assistant = True
            else:
                lines.append(f"{role}: {content}")
        text = "\n".join(lines)
        if add_generation_prompt and not has_assistant:
            text += "\nassistant: "
        return text

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Any:
        text = self._render(messages, add_generation_prompt)
        if not tokenize:
            return text
        ids = [self._id_of(token) for token in self._tokenize(text)]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def __call__(
        self, text: str, add_special_tokens: bool = False, **_: Any
    ) -> Dict[str, List[int]]:
        ids = [self._id_of(token) for token in self._tokenize(text)]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}

    def decode(
        self, token_ids: List[int], skip_special_tokens: bool = True
    ) -> str:
        pieces = []
        for token_id in token_ids:
            token = self._ids.get(int(token_id), "<unk>")
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return " ".join(pieces)


class _FakeLoraHead(nn.Module):
    """The trainable 'adapter': one hidden->vocab linear layer."""

    def __init__(self, hidden: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return self.linear(x)


class FakeWorkerModel(nn.Module):
    """Tiny model with a frozen base and a trainable adapter head.

    The embedding and the hidden linear form the frozen base (their hash
    is checked before/after training); ``lora_head`` is trainable and is
    what the optimizer updates. ``.model`` points at ``self`` because the
    trainer uses ``model.model.eval()/train()/named_parameters()``.
    """

    def __init__(
        self,
        tokenizer: FakeWorkerTokenizer,
        hidden: int = 8,
        device_name: str = "cpu",
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.device_name = device_name
        self.vocab_size = len(tokenizer.vocab)
        self.hidden = hidden
        self.base_embed = nn.Embedding(self.vocab_size, hidden)
        self.base_linear = nn.Linear(hidden, hidden)
        self.lora_head = _FakeLoraHead(hidden, self.vocab_size)
        for param in self.base_embed.parameters():
            param.requires_grad = False
        for param in self.base_linear.parameters():
            param.requires_grad = False
        # The trainer calls model.model.eval() / train() / named_parameters().
        self.model = self
        self.model_name = "fake/worker-model"
        self._loaded_adapter = False
        self.chat_template_sha256 = hashlib.sha256(
            tokenizer.chat_template.encode("utf-8")
        ).hexdigest()
        self._frozen_audit = {
            "base_frozen": True,
            "unfrozen_base": [],
            "trainable_names": sorted(
                n for n, p in self.named_parameters() if p.requires_grad
            ),
        }

    # ------------------------------------------------------------------
    # Forward / logits
    # ------------------------------------------------------------------

    def _forward(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        del attention_mask  # the fake ignores padding
        embedded = self.base_embed(input_ids)
        hidden = torch.relu(self.base_linear(embedded))
        return self.lora_head(hidden)

    def logits(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Logits [B, T, V] with autograd (teacher forcing)."""
        return self._forward(
            input_ids.to(self.device_name),
            attention_mask.to(self.device_name),
        )

    # ------------------------------------------------------------------
    # Trainer protocol
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> List[Tensor]:
        return [p for p in self.parameters() if p.requires_grad]

    def base_parameters(self) -> List[Tensor]:
        return [p for p in self.parameters() if not p.requires_grad]

    def parameter_norm_summary(self) -> Dict[str, Any]:
        per_module: Dict[str, float] = {}
        total = 0.0
        count = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            per_module[name] = float(param.detach().norm().item())
            total += per_module[name] ** 2
            count += param.numel()
        return {
            "trainable_parameters": len(per_module),
            "trainable_numel": count,
            "total_l2_norm": total**0.5,
            "per_parameter_l2": per_module,
        }

    def trainable_snapshot(self) -> Dict[str, Tensor]:
        return {
            name: param.detach().clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

    def trainable_delta_norm(self, snapshot: Dict[str, Tensor]) -> float:
        total = 0.0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            delta = param.detach() - snapshot[name].to(param.device)
            total += float(delta.norm().item()) ** 2
        return total**0.5

    def base_param_hash(self, sample_every: Optional[int] = None) -> str:
        hasher = hashlib.sha256()
        count = 0
        for name, param in self.named_parameters():
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
        """Deterministic greedy generation, same contract as the real one."""
        if seed is not None:
            torch.manual_seed(seed)
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
            current = ids
            for _ in range(max_new_tokens):
                logits = self._forward(current)
                next_id = int(logits[:, -1, :].argmax(dim=-1).item())
                generated.append(next_id)
                if next_id == int(eos):
                    break
                current = torch.cat(
                    [
                        current,
                        torch.tensor(
                            [[next_id]], device=self.device_name,
                            dtype=torch.long,
                        ),
                    ],
                    dim=1,
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
        path = Path(str(directory))
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "trainable": {
                    name: param.detach().cpu()
                    for name, param in self.named_parameters()
                    if param.requires_grad
                },
                "fake_model": True,
            },
            path / "adapter.pt",
        )

    def load_adapter(self, directory: Any) -> None:
        """Resume path only, mirroring the real wrapper's contract."""
        path = Path(str(directory)) / "adapter.pt"
        saved = torch.load(path, map_location="cpu")
        state = saved["trainable"]
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in state:
                    param.copy_(state[name])
        self._loaded_adapter = True

    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device_name,
            "dtype": "float32",
            "dtype_reason": "fake model: float32",
            "peft_r": "fake",
            "peft_alpha": "fake",
            "peft_dropout": 0.0,
            "peft_target_modules": None,
            "resolved_lora_modules": ["lora_head"],
            "frozen_audit": self._frozen_audit,
            "chat_template_sha256": self.chat_template_sha256,
            "max_input_tokens": 8192,
            "fresh_adapter": True,
            "loaded_previous_adapter": False,
        }


def build_fake_model(
    corpus: Optional[List[str]] = None,
    hidden: int = 8,
    device_name: str = "cpu",
) -> FakeWorkerModel:
    """One-call construction for tests."""
    tokenizer = FakeWorkerTokenizer(corpus=corpus)
    return FakeWorkerModel(tokenizer, hidden=hidden, device_name=device_name)
