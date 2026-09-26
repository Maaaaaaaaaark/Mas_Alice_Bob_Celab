"""Question selection tests (spec sec. 18.14): fixed 100-question filter,
deterministic alphabetical partition, manifest roundtrip + integrity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hotpot_mas.question_selection import (
    build_manifest,
    is_valid_candidate,
    load_manifest,
    select_questions,
)


def make_row(
    qid: str,
    answer: str = "Answer Text",
    q_type: str = "bridge",
    supporting: List[List] = None,
    context: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "_id": qid,
        "question": f"Question text for {qid}",
        "answer": answer,
        "type": q_type,
        "supporting_facts": supporting
        if supporting is not None
        else [["Doc A", 0], ["Doc B", 0]],
        "context": context
        if context is not None
        else [
            {"title": "Doc A", "sentences": ["Alpha sentence one.", "Alpha sentence two."]},
            {"title": "Doc B", "sentences": ["Beta sentence one.", "Beta sentence two."]},
            {"title": "Distractor", "sentences": ["Irrelevant distractor."]},
        ],
    }


class TestValidity:
    def test_valid_row_passes(self):
        assert is_valid_candidate(make_row("q1"))

    def test_empty_answer_rejected(self):
        assert not is_valid_candidate(make_row("q1", answer=""))

    def test_non_string_answer_rejected(self):
        assert not is_valid_candidate(make_row("q1", answer=42))  # type: ignore

    def test_wrong_type_rejected(self):
        assert not is_valid_candidate(make_row("q1", q_type="comparison_plus"))
        assert not is_valid_candidate(make_row("q1", q_type="intersection"))

    def test_single_supporting_title_rejected(self):
        assert not is_valid_candidate(
            make_row("q1", supporting=[["Doc A", 0], ["Doc A", 1]])
        )

    def test_three_supporting_titles_rejected(self):
        row = make_row("q1")
        row["context"].append(
            {"title": "Doc C", "sentences": ["Gamma sentence."]}
        )
        row["supporting_facts"] = [["Doc A", 0], ["Doc B", 0], ["Doc C", 0]]
        assert not is_valid_candidate(row)

    def test_title_not_in_context_rejected(self):
        assert not is_valid_candidate(
            make_row("q1", supporting=[["Doc A", 0], ["Ghost Doc", 0]])
        )

    def test_empty_supporting_paragraph_rejected(self):
        row = make_row("q1")
        for doc in row["context"]:
            if doc["title"] == "Doc A":
                doc["sentences"] = []
        assert not is_valid_candidate(row)


class TestSelection:
    def test_deterministic_order_for_same_seed(self):
        rows = [make_row(f"q{i}") for i in range(20)]
        first = select_questions(rows, 10, selection_seed=0)
        second = select_questions(rows, 10, selection_seed=0)
        assert [q.question_id for q in first] == [q.question_id for q in second]

    def test_alphabetical_partition_alice_gets_first_title(self):
        rows = [make_row("q1")]
        selected = select_questions(rows, 1, selection_seed=0)[0]
        assert selected.supporting_titles == ["Doc A", "Doc B"]
        assert selected.evidence_alice.startswith("Title: Doc A\n")
        assert selected.evidence_bob.startswith("Title: Doc B\n")
        # Evidence contains the full paragraph of the supporting doc.
        assert "Alpha sentence one." in selected.evidence_alice
        assert "Beta sentence two." in selected.evidence_bob
        # Distractors are never included.
        assert "Irrelevant" not in selected.evidence_alice
        assert "Irrelevant" not in selected.evidence_bob

    def test_distinct_supporting_docs_never_share_a_worker(self):
        rows = [make_row(f"q{i}") for i in range(15)]
        selected = select_questions(rows, 10, selection_seed=0)
        for q in selected:
            assert len(set(q.supporting_titles)) == 2
            alice_titles = q.evidence_alice.split("\n", 1)[0]
            bob_titles = q.evidence_bob.split("\n", 1)[0]
            assert alice_titles != bob_titles


class TestManifest:
    def test_roundtrip_and_integrity(self, tmp_path: Path):
        rows = [make_row(f"q{i}") for i in range(30)]
        path = tmp_path / "question_manifest.json"
        manifest = build_manifest(rows, 10, selection_seed=0, manifest_path=path)
        assert manifest["num_questions_selected"] == 10
        assert path.is_file()

        loaded = load_manifest(path)
        assert len(loaded) == 10
        assert [q.question_id for q in loaded] == [
            q["question_id"] for q in manifest["questions"]
        ]

    def test_tampered_manifest_fails_integrity_check(self, tmp_path: Path):
        rows = [make_row(f"q{i}") for i in range(20)]
        path = tmp_path / "question_manifest.json"
        build_manifest(rows, 10, selection_seed=0, manifest_path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["questions"][0]["question"] = "TAMPERED"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(RuntimeError, match="integrity check failed"):
            load_manifest(path)

    def test_insufficient_candidates_raises(self, tmp_path: Path):
        rows = [make_row(f"q{i}") for i in range(3)]
        with pytest.raises(RuntimeError, match="only 3 valid candidates"):
            build_manifest(rows, 10, selection_seed=0, manifest_path=tmp_path / "m.json")

    def test_manifest_records_excluded_statistics(self, tmp_path: Path):
        rows = [make_row(f"q{i}") for i in range(5)]
        rows.append(make_row("bad-answer", answer=""))
        rows.append(make_row("bad-type", q_type="intersection"))
        path = tmp_path / "question_manifest.json"
        manifest = build_manifest(rows, 5, selection_seed=0, manifest_path=path)
        stats = manifest["excluded_statistics"]
        assert stats.get("answer_missing_or_empty") == 1
        assert stats.get("type_not_bridge_or_comparison") == 1
