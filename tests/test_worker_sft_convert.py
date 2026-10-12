"""Conversion tests for Stage 1a worker SFT (SQuAD 2.0, MuSiQue,
reviewed/teacher entries). CPU-only, no model, no GPU."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from hotpot_mas.training.worker_sft.convert import (
    SPLITTER_VERSION,
    EvidenceSentence,
    WorkerDocument,
    _evidence_sentences_for_answer,
    _locate_reviewed_evidence,
    convert_musique_row,
    convert_squad2_row,
    load_reviewed_file,
    load_teacher_file,
    resolve_reviewed_entry,
    resolve_subquestion,
    split_sentences,
)


def make_squad_row(
    row_id: str,
    question: str,
    context: str,
    answers: List[str],
) -> Dict[str, Any]:
    starts = [context.find(answer) for answer in answers]
    return {
        "id": row_id,
        "question": question,
        "context": context,
        "answers": {"text": answers, "answer_start": starts},
    }


def test_convert_squad2_positive_uses_answer_containing_sentence():
    row = make_squad_row(
        "sq1",
        "When did the band form?",
        "The band started in 1990. It released many albums later.",
        ["1990"],
    )
    (example,) = convert_squad2_row(row, {"dataset": "squad_v2"})
    assert example.answerable
    assert example.source == "squad2"
    assert example.parent_id == "sq1"
    assert example.answer == "1990"
    assert len(example.documents) == 1
    assert example.documents[0].text == row["context"]  # original, untouched
    assert len(example.evidence) == 1
    assert example.evidence[0].sentence == "The band started in 1990."
    assert not example.excluded
    assert example.provenance["answer_offsets"] == (
        row["context"].find("1990"),
        row["context"].find("1990") + 4,
    )


def test_convert_squad2_positive_prepends_pronoun_sentence():
    row = make_squad_row(
        "sq2",
        "When did it happen?",
        "The fire broke out at dawn. It happened in 1844.",
        ["1844"],
    )
    (example,) = convert_squad2_row(row, {"dataset": "squad_v2"})
    assert [ev.sentence for ev in example.evidence] == [
        "The fire broke out at dawn.",
        "It happened in 1844.",
    ]


def test_convert_squad2_negative_is_official_unanswerable():
    row = make_squad_row(
        "sq3", "What color is it?", "Some context without an answer.", []
    )
    (example,) = convert_squad2_row(row, {"dataset": "squad_v2"})
    assert not example.answerable
    assert example.answer == ""
    assert example.negative_reason
    assert example.evidence == []


def test_convert_squad2_positive_with_missing_answer_is_excluded():
    row = make_squad_row(
        "sq4", "What year?", "Text without the answer string.", ["2020"]
    )
    (example,) = convert_squad2_row(row, {"dataset": "squad_v2"})
    assert example.excluded
    assert "conversion" in example.exclusion_reason


def test_resolve_subquestion_replaces_prior_answers_only():
    resolved, substitutions = resolve_subquestion(
        "When was #1 birthed?", ["East of England"]
    )
    assert resolved == "When was East of England birthed?"
    assert substitutions == [
        {
            "placeholder": "#1",
            "replacement": "East of England",
            "source": "answer of subquestion 0",
        }
    ]


def test_resolve_subquestion_never_inserts_current_answer():
    # prior_answers holds only EARLIER subquestions' answers; the current
    # subquestion's own answer cannot appear.
    resolved, _ = resolve_subquestion("Where is #1 located?", ["Paris"])
    assert resolved == "Where is Paris located?"
    assert "France" not in resolved


def test_resolve_subquestion_leaves_unresolvable_reference():
    resolved, substitutions = resolve_subquestion("Where is #9?", ["Paris"])
    assert resolved == "Where is #9?"
    assert substitutions == []


def _musique_row() -> Dict[str, Any]:
    return {
        "id": "mq-parent",
        "question": "Where was East of England birthed?",
        "question_decomposition": [
            {
                "question": "What is East of England called?",
                "answer": "East of England",
                "paragraph_support_idx": 0,
            },
            {
                "question": "When was #1 birthed?",
                "answer": "1994",
                "paragraph_support_idx": 1,
            },
        ],
        "paragraphs": [
            {
                "title": "Region",
                "paragraph_text": "The East of England is one of nine regions.",
            },
            {
                "title": "Created",
                "paragraph_text": "It was created in 1994 as a government region.",
            },
            {
                "title": "Sibling",
                "paragraph_text": "Another region was created in 2000.",
            },
        ],
    }


def test_convert_musique_row_resolves_refs_and_builds_sibling_negative():
    examples = convert_musique_row(_musique_row(), {"dataset": "musique"})
    positives = [e for e in examples if e.answerable]
    negatives = [e for e in examples if not e.answerable]
    assert len(positives) == 2
    sub1 = positives[0]
    assert sub1.local_question == "What is East of England called?"
    assert sub1.answer == "East of England"
    assert sub1.documents[0].title == "Region"
    sub2 = positives[1]
    assert sub2.local_question == "When was East of England birthed?"
    # The second subquestion resolves #1 with the FIRST answer only.
    assert sub2.substitutions[0]["replacement"] == "East of England"
    # All variants of the parent share the parent id -> one split.
    assert {e.parent_id for e in examples} == {"mq-parent"}
    # One sibling negative: re-asks a subquestion against a paragraph
    # that does not contain its answer.
    assert len(negatives) == 1
    neg = negatives[0]
    assert neg.answerable is False
    assert neg.negative_reason
    assert neg.documents[0].title == "Sibling"
    # The negative's local question uses the same resolved question and
    # the same recorded substitutions as its positive twin.
    assert neg.local_question in {sub1.local_question, sub2.local_question}
    if neg.local_question == sub2.local_question:
        assert neg.substitutions == sub2.substitutions


def test_split_sentences_splits_on_boundary_punctuation():
    text = "First sentence. Second one! Third? \"Quoted.\" Done."
    sentences = split_sentences(text)
    assert sentences[0] == "First sentence."
    assert sentences[1] == "Second one!"
    assert '"Quoted."' in sentences


def test_evidence_sentences_for_answer_missing_answer_fails():
    doc = WorkerDocument(
        doc_id="d0", title="", text="No answer here.", normalized_text=""
    )
    evidence, failed = _evidence_sentences_for_answer(doc, "42")
    assert evidence == []
    assert failed and "not found" in failed


# --------------------------------------------------------------------------
# Reviewed / teacher entries
# --------------------------------------------------------------------------

def make_reviewed_entry(**overrides: Any) -> Dict[str, Any]:
    entry = {
        "example_id": "reviewed-sq-1",
        "source": "squad2",
        "pointer": {"row_id": "sq-row-1"},
        "split": "train",
        "local_question": "When did the band form?",
        "original_question": "When did the band form?",
        "answer": "1990",
        "answer_aliases": [],
        "answerable": True,
        "documents": [],
        "evidence": [
            {"doc_id": "d0", "sentence": "The band started in 1990."}
        ],
        "substitutions": [],
        "review": {"status": "reviewed", "method": "fixture"},
    }
    entry.update(overrides)
    return entry


SQUAD_ROW = make_squad_row(
    "sq-row-1",
    "When did the band form?",
    "The band started in 1990. It released many albums later.",
    ["1990"],
)


def test_resolve_reviewed_entry_verifies_evidence_and_marks_reviewed():
    loaders = {"squad2": lambda pointer: SQUAD_ROW}
    example, problem = resolve_reviewed_entry(make_reviewed_entry(), loaders)
    assert problem is None
    assert example is not None
    assert example.review["status"] == "reviewed"
    assert example.evidence[0].sentence == "The band started in 1990."
    assert example.evidence[0].char_span is not None
    assert example.provenance["reviewed"] is True


def test_resolve_reviewed_entry_rejects_mismatched_evidence():
    loaders = {"squad2": lambda pointer: SQUAD_ROW}
    example, problem = resolve_reviewed_entry(
        make_reviewed_entry(
            evidence=[{"doc_id": "d0", "sentence": "Not in the context."}]
        ),
        loaders,
    )
    assert example is None
    assert problem and "not found verbatim" in problem


def test_resolve_reviewed_entry_rejects_evidence_without_answer():
    loaders = {"squad2": lambda pointer: SQUAD_ROW}
    example, problem = resolve_reviewed_entry(
        make_reviewed_entry(
            evidence=[
                {"doc_id": "d0", "sentence": "It released many albums later."}
            ]
        ),
        loaders,
    )
    assert example is None
    assert problem and "does not contain the answer" in problem


def test_resolve_reviewed_entry_reports_missing_official_row():
    loaders = {"squad2": lambda pointer: (_ for _ in ()).throw(KeyError("x"))}
    example, problem = resolve_reviewed_entry(make_reviewed_entry(), loaders)
    assert example is None
    assert problem and "not found" in problem


def test_locate_reviewed_evidence_allows_two_consecutive_sentences():
    doc = WorkerDocument(
        doc_id="d0",
        title="",
        text="The region is large. It was created in 1994. Other facts.",
        normalized_text="",
    )
    evidence, failed = _locate_reviewed_evidence(
        doc,
        ["The region is large.", "It was created in 1994."],
        "1994",
        [],
    )
    assert failed is None
    assert [ev.sentence for ev in evidence] == [
        "The region is large.",
        "It was created in 1994.",
    ]


def test_load_reviewed_file_checks_splitter_version(tmp_path):
    path = tmp_path / "reviewed.json"
    path.write_text(
        json.dumps(
            {"splitter_version": "old-version", "entries": []}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="splitter version"):
        load_reviewed_file(path, {})


def test_load_reviewed_file_collects_problems_not_silently(tmp_path):
    path = tmp_path / "reviewed.json"
    path.write_text(
        json.dumps(
            {
                "splitter_version": SPLITTER_VERSION,
                "entries": [
                    make_reviewed_entry(),
                    make_reviewed_entry(
                        example_id="reviewed-sq-2",
                        pointer={"row_id": "missing-row"},
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )
    loaders = {"squad2": lambda pointer: SQUAD_ROW}
    examples, problems = load_reviewed_file(path, loaders)
    assert [e.example_id for e in examples] == ["reviewed-sq-1"]
    assert [p["example_id"] for p in problems] == ["reviewed-sq-2"]
    assert "missing-row" in problems[0]["reason"]


def test_load_teacher_file_reports_missing_file(tmp_path):
    entries, status = load_teacher_file(tmp_path / "absent.jsonl")
    assert entries == []
    assert status == {
        "status": "missing",
        "path": str(tmp_path / "absent.jsonl"),
    }


def test_load_teacher_file_loads_jsonl_entries(tmp_path):
    path = tmp_path / "teacher.jsonl"
    path.write_text(
        json.dumps({"example_id": "t1"}) + "\n" + json.dumps({"example_id": "t2"}) + "\n",
        encoding="utf-8",
    )
    entries, status = load_teacher_file(path)
    assert [e["example_id"] for e in entries] == ["t1", "t2"]
    assert status["status"] == "loaded"


def test_resolve_hotpot_reviewed_entry_reads_official_context_shape():
    row = {
        "id": "hp-row-1",
        "question": "When did it begin?",
        "context": {
            "title": ["Relevant", "Distractor"],
            "sentences": [
                ["The magazine began in 1989.", "It is still published."],
                ["This paragraph is unrelated."],
            ],
        },
    }
    entry = {
        "example_id": "reviewed-hp-1",
        "source": "hotpot",
        "pointer": {"row_id": "hp-row-1", "titles": ["Relevant"]},
        "split": "train",
        "local_question": "When did the magazine begin?",
        "original_question": "When did it begin?",
        "answer": "1989",
        "answer_aliases": [],
        "answerable": True,
        "documents": [{"doc_id": "d0", "title": "Relevant"}],
        "evidence": [
            {"doc_id": "d0", "sentence": "The magazine began in 1989."}
        ],
        "substitutions": [],
        "review": {"status": "reviewed", "method": "fixture"},
    }
    example, problem = resolve_reviewed_entry(
        entry, {"hotpot": lambda pointer: row}
    )
    assert problem is None
    assert example is not None
    assert example.documents[0].title == "Relevant"
    assert example.evidence[0].sentence == "The magazine began in 1989."
