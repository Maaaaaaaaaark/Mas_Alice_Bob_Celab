"""Parser unit tests: routing, termination, uniqueness, no-guess behavior."""

from __future__ import annotations

from hotpot_mas.parser import extract_final_answer, parse_celab_output


def test_ask_alice():
    result = parse_celab_output("Celab: <TO>ALICE</TO> please tell me X")
    assert result.status == "ok"
    assert result.action == "ask_alice"
    assert result.body is None
    assert result.error is None


def test_ask_bob():
    result = parse_celab_output("Celab: <TO>BOB</TO> please tell me Y")
    assert result.status == "ok"
    assert result.action == "ask_bob"


def test_final_extracts_inner_text():
    result = parse_celab_output("Celab: <FINAL>The Connector Bridge</FINAL>")
    assert result.status == "ok"
    assert result.action == "final"
    assert result.body == "The Connector Bridge"


def test_final_body_is_stripped():
    result = parse_celab_output("Celab: <FINAL>  padded answer  </FINAL>")
    assert result.body == "padded answer"


def test_empty_final_body_is_valid_empty_answer():
    # Design choice: <FINAL></FINAL> is a valid final with an empty
    # prediction (evaluated as empty), distinct from a parse error (null).
    result = parse_celab_output("Celab: <FINAL></FINAL>")
    assert result.status == "ok"
    assert result.action == "final"
    assert result.body == ""


def test_final_matches_across_newlines():
    result = parse_celab_output("Celab: <FINAL>line one\nline two</FINAL>")
    assert result.status == "ok"
    assert result.body == "line one\nline two"


def test_unclosed_final_uses_logged_fallback_to_end_of_output():
    result = parse_celab_output("Celab: <FINAL>October 1922")
    assert result.status == "ok_unclosed_final_fallback"
    assert result.action == "final"
    assert result.body == "October 1922"
    assert result.error is None


def test_unclosed_empty_final_is_valid_empty_fallback():
    result = parse_celab_output("Celab: <FINAL>   ")
    assert result.status == "ok_unclosed_final_fallback"
    assert result.action == "final"
    assert result.body == ""


def test_unclosed_final_mixed_with_route_is_ambiguous():
    result = parse_celab_output(
        "Celab: <TO>ALICE</TO> ask first <FINAL>premature answer"
    )
    assert result.status == "error"
    assert result.action is None


def test_repeated_unclosed_final_is_ambiguous():
    result = parse_celab_output("Celab: <FINAL>one <FINAL>two")
    assert result.status == "error"
    assert result.action is None


def test_orphan_final_close_is_not_accepted():
    result = parse_celab_output("Celab: answer</FINAL>")
    assert result.status == "error"
    assert result.action is None


def test_no_marker_is_parse_error():
    result = parse_celab_output("Celab: I have no idea what to do")
    assert result.status == "error"
    assert result.action is None
    assert "no action marker" in (result.error or "")


def test_two_different_markers_is_parse_error():
    result = parse_celab_output(
        "Celab: <TO>ALICE</TO> x <FINAL>answer</FINAL>"
    )
    assert result.status == "error"
    assert result.action is None


def test_repeated_same_marker_is_ambiguous_parse_error():
    # Design choice: the same marker twice is ambiguous, not a guess.
    result = parse_celab_output(
        "Celab: <TO>ALICE</TO> x <TO>ALICE</TO> y"
    )
    assert result.status == "error"
    assert result.action is None
    assert "2 times" in (result.error or "")


def test_incomplete_marker_does_not_match():
    # "<TO>ALICE" without the closing tag is not a marker.
    result = parse_celab_output("Celab: <TO>ALICE")
    assert result.status == "error"


def test_markers_are_case_sensitive():
    result = parse_celab_output("Celab: <to>alice</to> please")
    assert result.status == "error"


def test_extract_final_answer_accepts_unique_final():
    result = extract_final_answer("Celab: <FINAL>x</FINAL>")
    assert result.status == "ok"
    assert result.body == "x"


def test_extract_final_answer_accepts_unclosed_fallback():
    result = extract_final_answer("Celab: <FINAL>x")
    assert result.status == "ok_unclosed_final_fallback"
    assert result.body == "x"


def test_extract_final_answer_rejects_to_marker():
    result = extract_final_answer("Celab: <TO>BOB</TO> more info")
    assert result.status == "error"
    assert result.action is None
