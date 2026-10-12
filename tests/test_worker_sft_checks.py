"""Automatic label checks + rendered target format checks for Stage 1a."""

from __future__ import annotations

from typing import Any, Dict, List

from hotpot_mas.training.worker_sft.checks import (
    checked_example,
    example_matches_target_format,
    is_reviewed,
    normalize_question_key,
    passes_all,
    run_automatic_checks,
)

from worker_sft_test_utils import make_doc, make_evidence, make_example


def _positive() -> Any:
    doc = make_doc("d0", "The tower opened in 1889. It stands in Paris.")
    return make_example(
        "pos-1",
        "When did the tower open?",
        [doc],
        answer="1889",
        evidence=[make_evidence("d0", "The tower opened in 1889.")],
    )


def _negative() -> Any:
    doc = make_doc("d0", "The museum shows modern art daily.")
    return make_example(
        "neg-1",
        "When was the museum founded?",
        [doc],
        answer="1920",
        answerable=False,
        negative_reason="official unanswerable",
    )


def test_positive_example_passes_all_checks():
    results = run_automatic_checks(_positive())
    assert all(info["ok"] for info in results.values())


def test_negative_example_passes_all_checks():
    results = run_automatic_checks(_negative())
    assert all(info["ok"] for info in results.values())


def test_evidence_not_verbatim_in_doc_fails():
    example = _positive()
    example.evidence[0].sentence = "The tower opened in 1899."
    results = run_automatic_checks(example)
    assert not results["evidence_in_docs"]["ok"]
    assert results["source_ids_valid"]["ok"]


def test_char_span_mismatch_fails_offsets_consistent():
    example = _positive()
    example.evidence[0].char_span = (0, 5)  # wrong window
    results = run_automatic_checks(example)
    assert not results["offsets_consistent"]["ok"]


def test_evidence_without_answer_fails_answer_covered():
    example = _positive()
    example.evidence[0].sentence = "It stands in Paris."
    example.evidence[0].char_span = (
        example.documents[0].text.find("It stands in Paris."),
        example.documents[0].text.find("It stands in Paris.") + 17,
    )
    results = run_automatic_checks(example)
    assert not results["answer_covered"]["ok"]


def test_answer_in_visible_docs_flags_negative_for_semantic_review():
    example = _negative()
    example.documents[0] = make_doc("d0", "It was founded in 1920.")
    example.documents[0].normalized_text = "it was founded in 1920"
    results = run_automatic_checks(example)
    screen = results["answer_string_screen"]
    assert screen["ok"]
    assert screen["flagged"]
    assert screen["present_in"] == ["d0"]
    assert "review the entity/relation" in screen["reason"]


def test_negative_without_reason_fails():
    example = _negative()
    example.negative_reason = None
    results = run_automatic_checks(example)
    assert not results["negative_reason_present"]["ok"]


def test_unresolved_ref_in_question_fails():
    example = _positive()
    example.local_question = "When was #3 built?"
    results = run_automatic_checks(example)
    assert not results["no_unresolved_refs"]["ok"]


def test_checked_example_excludes_with_recorded_reason():
    example = _positive()
    example.evidence[0].sentence = "Not in the document."
    checked, failures = checked_example(example)
    assert checked.excluded
    assert checked.exclusion_reason.startswith(
        "failed automatic label checks"
    )
    assert any("evidence_in_docs" in failure for failure in failures)


def test_auto_checks_never_set_reviewed_status():
    example = _positive()
    example.review = {"status": "auto_checked"}
    checked_example(example)
    assert not is_reviewed(example)
    assert example.review["status"] == "auto_checked"


def test_reviewed_status_is_independent_of_checks():
    example = _positive()
    assert is_reviewed(example)  # fixture default is reviewed
    example2 = _positive()
    example2.review["status"] = "auto_checked"
    assert not is_reviewed(example2)
    assert passes_all(example2)  # checks pass either way


def test_target_format_accepts_one_and_two_lines():
    example = _positive()
    ok, reason = example_matches_target_format(
        example, 'D0: "The tower opened in 1889."'
    )
    assert ok, reason
    ok, reason = example_matches_target_format(
        example,
        'D0: "The tower opened in 1889."\nD0: "It stands in Paris."',
    )
    assert ok, reason


def test_target_format_rejects_unknown_display_id():
    example = _positive()  # one document -> only D0 exists
    ok, reason = example_matches_target_format(
        example, 'D5: "The tower opened in 1889."'
    )
    assert not ok
    assert "only 1 documents" in reason


def test_target_format_rejects_more_than_two_lines():
    example = _positive()
    target = "\n".join(
        [f'D0: "Sentence {i}."' for i in range(3)]
    )
    ok, reason = example_matches_target_format(example, target)
    assert not ok
    assert "more than two lines" in reason


def test_target_format_requires_exact_not_found_for_negatives():
    example = _negative()
    ok, _ = example_matches_target_format(example, "NOT FOUND")
    assert ok
    ok, reason = example_matches_target_format(example, "not found")
    assert not ok
    ok, reason = example_matches_target_format(
        example, 'D0: "The museum shows modern art daily."'
    )
    assert not ok


def test_target_format_rejects_bad_line_shape():
    example = _positive()
    ok, reason = example_matches_target_format(
        example, "D0: The tower opened in 1889."
    )
    assert not ok
    ok, reason = example_matches_target_format(example, 'D0: ""')
    assert not ok
    assert "empty" in reason


def test_normalize_question_key_matches_normalization_contract():
    assert normalize_question_key("When  did IT happen?!") == (
        "when did it happen"
    )
