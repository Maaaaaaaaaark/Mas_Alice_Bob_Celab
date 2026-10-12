"""Focused tests for the real wrapper's generation loop without HF downloads."""

from types import SimpleNamespace

import torch

from hotpot_mas.training.worker_sft.model import HFLoraWorkerModel


class _Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return torch.tensor([[1, 2]], dtype=torch.long)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(value) for value in ids if value != 9)


class _CachedModel:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        self.calls.append((input_ids.detach().clone(), past_key_values))
        next_id = 7 if len(self.calls) == 1 else 9
        logits = torch.full((1, input_ids.shape[1], 10), -100.0)
        logits[:, -1, next_id] = 100.0
        return SimpleNamespace(logits=logits, past_key_values="cache")


def test_generation_passes_cache_after_first_token():
    wrapper = HFLoraWorkerModel.__new__(HFLoraWorkerModel)
    wrapper.device_name = "cpu"
    wrapper.tokenizer = _Tokenizer()
    wrapper.model = _CachedModel()
    wrapper._stop_token_ids = [9]
    output = wrapper.generate(
        [{"role": "user", "content": "question"}], max_new_tokens=4
    )
    assert output["new_token_ids"] == [7, 9]
    assert wrapper.model.calls[0][1] is None
    assert wrapper.model.calls[1][1] == "cache"
    assert wrapper.model.calls[1][0].tolist() == [[7]]
