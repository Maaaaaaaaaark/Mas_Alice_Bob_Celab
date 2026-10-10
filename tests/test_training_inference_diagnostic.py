"""CPU-only checks for the inference diagnostic's data and metrics."""

from __future__ import annotations

from hotpot_mas.question_selection import SelectedQuestion
from hotpot_mas.training.inference_diagnostic import (
    CONDITIONS,
    gold_supporting_reports,
    literal_string_present,
    normalized_span_present,
    render_summary,
    summarize_records,
)


def _question() -> SelectedQuestion:
    return SelectedQuestion(
        question_id="q1",
        question="When did it end?",
        answer="October 1922",
        q_type="bridge",
        supporting_titles=["Alpha", "Zulu"],
        evidence_alice="unused",
        evidence_bob="unused",
        context_titles=["Zulu", "Noise", "Alpha"],
        evidence_all="all documents",
        document_pool=[
            {
                "title": "Zulu",
                "paragraph": "The war ended in October 1922.",
                "is_supporting": True,
            },
            {
                "title": "Noise",
                "paragraph": "Irrelevant.",
                "is_supporting": False,
            },
            {
                "title": "Alpha",
                "paragraph": "This identifies the war.",
                "is_supporting": True,
            },
        ],
    )


def test_gold_supporting_reports_follow_worker_title_order_without_labels():
    alice, bob = gold_supporting_reports(_question())
    assert alice.startswith("Title: Alpha")
    assert bob.startswith("Title: Zulu")
    assert "is_supporting" not in alice + bob


def test_normalized_span_presence_uses_hotpot_normalization():
    assert normalized_span_present("It ended in October, 1922.", "October 1922")
    assert normalized_span_present("The United States", "United States")
    assert not normalized_span_present("October was cold in 1922", "October 1922")
    assert literal_string_present("It ended in OCTOBER 1922.", "October 1922")
    assert not literal_string_present("October, 1922", "October 1922")


def test_summary_has_four_conditions_and_separate_worker_rates():
    records = []
    for condition in CONDITIONS:
        record = {
            "question_id": "q1",
            "condition": condition,
            "f1": 0.5,
            "em": 0.0,
            "parsed": True,
            "c_generated_tokens": 4,
            "normalized_answer_words": 2,
        }
        if condition == "untrained_worker_reports":
            record.update(
                alice_contains_gold_string=True,
                bob_contains_gold_string=False,
                alice_contains_normalized_gold=True,
                bob_contains_normalized_gold=False,
            )
        records.append(record)

    summary = summarize_records(records, expected_questions=1)
    assert summary["complete"] is True
    assert summary["conditions"]["empty_reports"]["mean_f1"] == 0.5
    containment = summary["untrained_worker_gold_answer_containment"]
    assert containment["alice_gold_string_rate"] == 1.0
    assert containment["bob_gold_string_rate"] == 0.0
    assert "全部 10 篇文档" in render_summary(summary)
