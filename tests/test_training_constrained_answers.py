from hotpot_mas.training.config import DecodeConfig, TrainingConfig
from hotpot_mas.training.synthesizers import HFSynthesizer


def test_decode_config_round_trips_answer_constraints():
    raw = {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 16,
        "answer_prefix": "Answer:",
        "stop_on_newline": True,
        "strip_answer_labels": True,
    }
    assert DecodeConfig.from_dict(raw).to_dict() == raw


def test_answer_parser_removes_known_leading_labels_and_new_lines():
    strip = HFSynthesizer._strip_answer_labels
    assert strip("Answer: October 1922\nextra") == "October 1922"
    assert strip("Celab: Answer: yes") == "yes"
    assert strip("Reader: no") == "no"


def test_constrained_diagnostic_config_is_short_and_greedy():
    cfg = TrainingConfig.from_yaml(
        "configs/inference_diagnostic_300_constrained.yaml"
    )
    decode = cfg.workers.synthesizer_decode
    assert decode.do_sample is False
    assert decode.max_new_tokens == 16
    assert decode.answer_prefix == "Answer:"
    assert decode.stop_on_newline is True
    assert decode.strip_answer_labels is True
