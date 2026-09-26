"""Evaluator tests: port of the official hotpot_evaluate_v1.py semantics."""

from __future__ import annotations

from hotpot_mas.evaluation import (
    evaluate_answer,
    exact_match_score,
    f1_score,
    normalize_answer,
)


def test_normalize_answer_lowercases_and_strips_articles():
    assert normalize_answer("The Beatles") == "beatles"
    assert normalize_answer("A Very Good Answer") == "very good answer"


def test_normalize_answer_removes_punctuation():
    assert normalize_answer("What's up?") == "whats up"
    assert normalize_answer("Paris, France (1889).") == "paris france 1889"


def test_normalize_answer_fixes_whitespace():
    assert normalize_answer("  spaced   out  ") == "spaced out"


def test_f1_partial_overlap():
    # Official normalization removes "the": precision=3/4, recall=3/3,
    # hence F1=6/7.
    assert abs(f1_score("This is my answer", "This is the answer") - 6 / 7) < 1e-9


def test_f1_no_overlap_is_zero():
    assert f1_score("saint", "river") == 0.0


def test_f1_exact_match_is_one():
    assert f1_score("Brooklyn Bridge", "Brooklyn Bridge") == 1.0


def test_f1_yes_no_guard():
    assert f1_score("yes", "no") == 0.0
    assert f1_score("yes", "yes") == 1.0
    assert f1_score("no", "noanswer") == 0.0
    assert f1_score("noanswer", "noanswer") == 1.0


def test_em_normalized_equality():
    assert exact_match_score("The Beatles", "the beatles") == 1.0
    assert exact_match_score("The Beatles", "The Rolling Stones") == 0.0


def test_evaluate_answer_none_is_empty_prediction():
    f1, em = evaluate_answer(None, "The Connector Bridge")
    assert f1 == 0.0
    assert em == 0.0


def test_evaluate_answer_empty_string():
    f1, em = evaluate_answer("", "x")
    assert f1 == 0.0
    assert em == 0.0


def test_evaluate_answer_single_gold():
    f1, em = evaluate_answer("The Connector Bridge", "The Connector Bridge")
    assert f1 == 1.0
    assert em == 1.0


def test_evaluate_answer_takes_max_over_gold_list():
    f1, em = evaluate_answer("The Connector Bridge", ["Golden Gate", "The Connector Bridge"])
    assert em == 1.0
    assert f1 == 1.0
