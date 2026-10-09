"""FakeTokenizer / FakePolicy / FakeSynthesizer offline doubles."""

from __future__ import annotations

import torch
import pytest

from hotpot_mas.training.config import DecodeConfig
from hotpot_mas.training.fake_policy import (
    FakePolicy,
    FakeSynthesizer,
    FakeTokenizer,
)
from tests.training_test_utils import GOLD_ANSWER, WRONG_ANSWER


@pytest.fixture
def policy() -> FakePolicy:
    return FakePolicy()


class TestFakeTokenizer:
    def test_roundtrip(self):
        tokenizer = FakeTokenizer()
        ids = tokenizer.encode("w1 w3 [eos]")
        assert ids == [1, 3, len(tokenizer.vocab) - 1]
        assert tokenizer.decode(ids) == "w1 w3 [eos]"

    def test_unknown_word_maps_to_zero(self):
        tokenizer = FakeTokenizer()
        assert tokenizer.encode("totally-new-word") == [0]

    def test_eos_id_is_last(self):
        tokenizer = FakeTokenizer()
        assert tokenizer.eos_token_id == len(tokenizer.vocab) - 1


class TestFakePolicy:
    def test_sampling_is_seeded_and_reproducible(self, policy: FakePolicy):
        decode = DecodeConfig(
            do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=8
        )
        first = policy.sample_report([], 42, decode)
        second = policy.sample_report([], 42, decode)
        assert first.token_ids == second.token_ids
        assert first.logprobs == second.logprobs

    def test_sampling_logprobs_match_theta(self, policy: FakePolicy):
        init = torch.linspace(0.0, 1.0, len(policy.tokenizer.vocab))
        policy.logits.data.copy_(init)
        decode = DecodeConfig(
            do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=5
        )
        report = policy.sample_report([], 7, decode)
        expected = torch.log_softmax(policy.logits.detach(), dim=0)
        for token, logp in zip(report.token_ids, report.logprobs):
            assert logp == pytest.approx(float(expected[token]))

    def test_teacher_force_has_grad(self, policy: FakePolicy):
        logp = policy.teacher_force([0, 1], [2, 3, 4])
        assert logp.requires_grad
        assert logp.shape == (3,)
        logp.sum().backward()
        assert policy.logits.grad is not None

    def test_teacher_force_ignores_prompt_ids(self, policy: FakePolicy):
        a = policy.teacher_force([0, 0, 0], [2, 3])
        b = policy.teacher_force([1, 1, 1, 1, 1], [2, 3])
        assert torch.allclose(a, b)

    def test_shared_theta_is_one_parameter(self, policy: FakePolicy):
        # A and B share the exact same parameter object: there is exactly
        # one trainable parameter regardless of how many "sides" use it.
        params = policy.trainable_parameters()
        assert len(params) == 1
        assert params[0] is policy.logits
        assert len(policy.state_dict()) == 1

    def test_state_dict_roundtrip(self, policy: FakePolicy):
        policy.logits.data.copy_(torch.arange(len(policy.tokenizer.vocab)).float())
        state = policy.state_dict()
        other = FakePolicy()
        other.load_state_dict(state)
        assert torch.allclose(other.logits, policy.logits)

    def test_deterministic_decode_report(self, policy: FakePolicy):
        decode = DecodeConfig(
            do_sample=False, temperature=0.0, top_p=1.0, max_new_tokens=4
        )
        first = policy.decode_report([], 1, decode)
        second = policy.decode_report([], 999, decode)
        assert first.token_ids == second.token_ids


class TestFakeSynthesizer:
    def test_answers_and_counts_calls(self):
        calls = []

        def answer_fn(q: str, a: str, b: str) -> str:
            calls.append((q, a, b))
            return GOLD_ANSWER

        synth = FakeSynthesizer(answer_fn)
        messages = [{"role": "user", "content": "Original question?"}]
        result = synth.answer(
            messages, GOLD_ANSWER, question="question?",
            a_report="report a", b_report="report b",
        )
        synth.answer(
            messages, GOLD_ANSWER, question="question?",
            a_report="report a", b_report="report b",
        )
        assert synth.call_count == 2
        assert calls == [
            ("question?", "report a", "report b"),
            ("question?", "report a", "report b"),
        ]
        # The fake runs the real parsing + F1 pipeline: an answer equal to
        # the gold string scores 1.0, and no <FINAL> tag means the raw
        # output is kept as the prediction (parsed=False).
        assert result.f1 == pytest.approx(1.0)
        assert result.pred_answer == GOLD_ANSWER
        assert result.parsed is False

    def test_final_tag_is_parsed_like_the_real_synthesizer(self):
        synth = FakeSynthesizer(
            lambda q, a, b: "<FINAL>gold answer text</FINAL>"
        )
        result = synth.answer([], GOLD_ANSWER)
        assert result.parsed is True
        assert result.pred_answer == "gold answer text"
        assert result.f1 == pytest.approx(1.0)

    def test_wrong_answer_scores_zero(self):
        synth = FakeSynthesizer(lambda q, a, b: WRONG_ANSWER)
        result = synth.answer([], GOLD_ANSWER)
        assert result.f1 == pytest.approx(0.0)

    def test_info_reports_frozen(self):
        synth = FakeSynthesizer(lambda q, a, b: "x")
        assert synth.info()["frozen"] is True
