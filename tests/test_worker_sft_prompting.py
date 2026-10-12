"""Prompt building, document shuffling + display-id remapping, and output
parsing/evaluation for Stage 1a. CPU-only."""

from __future__ import annotations

from typing import Any, Dict, List

from hotpot_mas.training.worker_sft.prompting import (
    NOT_FOUND,
    build_messages,
    evaluate_output,
    load_instruction,
    parse_output,
    render_target,
    render_user_message,
    shuffle_and_remap,
)

from worker_sft_test_utils import (
    PROMPTS_WORKER_SFT_DIR,
    make_doc,
    make_evidence,
    make_example,
)

INSTRUCTION = load_instruction(PROMPTS_WORKER_SFT_DIR)


def two_doc_example(**kw: Any) -> Any:
    d0 = make_doc("d0", "The Louvre opened in 1793.", "Louvre")
    d1 = make_doc("d1", "The Eiffel Tower opened in 1889.", "Tower")
    answer = kw.pop("answer", "1793")
    return make_example(
        "p1",
        "When did the Louvre open?",
        [d0, d1],
        answer=answer,
        evidence=[make_evidence("d0", "The Louvre opened in 1793.")],
        **kw,
    )


def test_shuffle_and_remap_is_deterministic_and_remaps_ids():
    example = two_doc_example()
    ordered_a, map_a = shuffle_and_remap(example, 42)
    ordered_b, map_b = shuffle_and_remap(example, 42)
    assert [d.doc_id for d in ordered_a] == [d.doc_id for d in ordered_b]
    assert map_a == map_b
    # The map is a bijection onto display positions.
    assert set(map_a.values()) == {0, 1}
    # Stable ids d0/d1 stay on the documents; display ids are positions.
    assert all(d.doc_id.startswith("d") for d in ordered_a)
    assert map_a["d0"] != map_a["d1"]


def test_render_target_uses_the_same_remapped_positions():
    example = two_doc_example()
    ordered, position_by_doc_id = shuffle_and_remap(example, 7)
    target = render_target(example, position_by_doc_id)
    expected_position = position_by_doc_id["d0"]
    assert target == f'D{expected_position}: "The Louvre opened in 1793."'


def test_user_message_contains_question_and_both_documents():
    example = two_doc_example()
    ordered, _ = shuffle_and_remap(example, 0)
    user = render_user_message(example, ordered)
    assert example.local_question in user
    assert "D0: " in user and "D1: " in user
    assert "The Louvre opened in 1793." in user
    assert "The Eiffel Tower opened in 1889." in user


def test_build_messages_never_includes_answer_metadata():
    example = two_doc_example(answer="1793", aliases=["seventeen ninety-three"])
    messages, target = build_messages(example, INSTRUCTION, 0)
    prompt = "\n".join(m["content"] for m in messages)
    # The prompt quotes the documents (which contain the answer string as
    # part of a sentence) but never the answer/alias fields themselves.
    assert "seventeen ninety-three" not in prompt
    assert target.startswith("D") and '"' in target


def test_negative_target_is_exact_not_found():
    example = make_example(
        "n1",
        "When was the museum founded?",
        [make_doc("d0", "The museum shows modern art daily.")],
        answer="1920",
        answerable=False,
        negative_reason="official unanswerable",
    )
    _, target = build_messages(example, INSTRUCTION, 0)
    assert target == NOT_FOUND


def test_parse_output_recognizes_not_found():
    parsed = parse_output("NOT FOUND")
    assert parsed["abstained"] and not parsed["malformed"]


def test_parse_output_recognizes_one_line_per_source():
    parsed = parse_output('D1: "Sentence one."\nD0: "Sentence two."')
    assert not parsed["malformed"]
    assert [line["position"] for line in parsed["lines"]] == [1, 0]


def test_parse_output_flags_malformed_lines():
    assert parse_output("D0: missing quotes").get("malformed")
    assert parse_output("some random text").get("malformed")
    assert parse_output("").get("malformed")


def test_parse_output_allows_quotes_inside_exact_sentence():
    parsed = parse_output(
        'D0: "It merged into \"Godey\'s Lady\'s Book\" in 1846."'
    )
    assert not parsed["malformed"]
    assert parsed["lines"][0]["quote"] == (
        'It merged into "Godey\'s Lady\'s Book" in 1846.'
    )


def test_evaluate_output_positive_is_correct_on_coverage():
    example = two_doc_example()
    ordered, _ = shuffle_and_remap(example, 0)
    result = evaluate_output(
        example, ordered, 'D0: "The Louvre opened in 1793."'
    )
    assert result["parse_ok"]
    assert result["source_valid"]
    assert result["quotes_verbatim"]
    assert result["answer_covered"]
    assert result["correct"]


def test_evaluate_output_positive_requires_verbatim_quote():
    example = two_doc_example()
    ordered, _ = shuffle_and_remap(example, 0)
    result = evaluate_output(
        example, ordered, 'D0: "The Louvre museum opened in 1793."'
    )
    assert not result["quotes_verbatim"]
    assert not result["correct"]


def test_evaluate_output_requires_a_complete_sentence_not_substring():
    example = two_doc_example()
    ordered, _ = shuffle_and_remap(example, 0)
    result = evaluate_output(example, ordered, 'D0: "opened in 1793"')
    assert not result["quotes_verbatim"]
    assert not result["correct"]


def test_evaluate_output_checks_source_positions():
    example = two_doc_example()
    ordered, _ = shuffle_and_remap(example, 0)
    result = evaluate_output(
        example, ordered, 'D3: "The Louvre opened in 1793."'
    )
    assert not result["source_valid"]
    assert not result["correct"]


def test_evaluate_output_negative_is_correct_only_on_abstention():
    example = make_example(
        "n1",
        "When was the museum founded?",
        [make_doc("d0", "The museum shows modern art daily.")],
        answer="1920",
        answerable=False,
        negative_reason="official unanswerable",
    )
    ordered, _ = shuffle_and_remap(example, 0)
    abstained = evaluate_output(example, ordered, "NOT FOUND")
    assert abstained["abstention_correct"] and abstained["correct"]
    answered = evaluate_output(
        example, ordered, 'D0: "The museum shows modern art daily."'
    )
    assert not answered["abstention_correct"]
    assert not answered["correct"]
    assert answered["answer_covered"] is False


def test_two_sentence_target_with_two_sources_round_trips():
    d0 = make_doc("d0", "Arthur's Magazine began in 1844.", "Arthur")
    d1 = make_doc("d1", "In May 1846 it was merged into Godey's.", "Godey")
    example = make_example(
        "p2",
        "When was Arthur's Magazine merged into Godey's?",
        [d0, d1],
        answer="May 1846",
        evidence=[
            make_evidence("d0", "Arthur's Magazine began in 1844."),
            make_evidence("d1", "In May 1846 it was merged into Godey's."),
        ],
    )
    ordered, positions = shuffle_and_remap(example, 3)
    target = render_target(example, positions)
    lines = [line for line in target.splitlines() if line.strip()]
    assert len(lines) == 2
    for line in lines:
        position = int(line.split(":")[0][1:])
        quote = line.split('"')[1]
        assert quote in ordered[position].text
