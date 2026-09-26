"""Unit tests for vLLM accounting that do not require vLLM or a GPU."""

from __future__ import annotations

from hotpot_mas.model_engine import _vllm_generated_token_count


def test_vllm_count_keeps_returned_eos_once():
    assert _vllm_generated_token_count(
        [10, 11, 106], "stop", None, [1, 106]
    ) == 3


def test_vllm_count_restores_omitted_eos():
    assert _vllm_generated_token_count(
        [10, 11], "stop", None, [1, 106]
    ) == 3


def test_vllm_count_restores_explicit_omitted_eos_reason():
    assert _vllm_generated_token_count(
        [10, 11], "stop", 106, [1, 106]
    ) == 3


def test_vllm_count_does_not_add_for_length_cap():
    assert _vllm_generated_token_count(
        [10, 11], "length", None, [1, 106]
    ) == 2


def test_vllm_count_does_not_treat_non_eos_stop_as_eos():
    assert _vllm_generated_token_count(
        [10, 11], "stop", "custom stop", [1, 106]
    ) == 2
