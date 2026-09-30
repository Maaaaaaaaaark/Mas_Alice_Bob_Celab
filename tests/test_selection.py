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


def make_hf_row(qid: str) -> Dict[str, Any]:
    """Current Hugging Face dict-of-parallel-lists representation."""
    return {
        "id": qid,
        "question": "Which document supplies the answer?",
        "answer": "Doc B",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {
            "title": ["Doc A", "Doc B"],
            "sent_id": [0, 1],
        },
        "context": {
            "title": ["Doc A", "Distractor", "Doc B"],
            "sentences": [
                ["Alpha."],
                ["Irrelevant."],
                ["Beta one.", "Beta two."],
            ],
        },
    }


def make_distractor_row(qid: str) -> Dict[str, Any]:
    row = make_row(qid)
    row["context"] = [
        {"title": "Doc A", "sentences": ["Gold alpha."]},
        {"title": "Doc B", "sentences": ["Gold beta."]},
        *[
            {
                "title": f"Distractor {index}",
                "sentences": [f"Irrelevant text {index}."],
            }
            for index in range(8)
        ],
    ]
    return row


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
    def test_current_huggingface_schema_and_id_are_supported(self):
        row = make_hf_row("hf-id-123")
        assert is_valid_candidate(row)
        selected = select_questions([row], 1, selection_seed=0)[0]
        assert selected.question_id == "hf-id-123"
        assert selected.supporting_facts == [["Doc A", 0], ["Doc B", 1]]
        assert selected.hotpotqa_metadata == {"level": "hard", "type": "bridge"}
        assert selected.partition_metadata["distractors_included"] is False
        assert "Irrelevant" not in selected.evidence_alice
        assert "Irrelevant" not in selected.evidence_bob

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

    def test_balanced_distractor_partition_is_disjoint_hidden_and_stable(self):
        row = make_distractor_row("q-balanced")
        first = select_questions(
            [row],
            1,
            selection_seed=0,
            evidence_partition="balanced_distractor",
            partition_seed=17,
        )[0]
        second = select_questions(
            [row],
            1,
            selection_seed=999,
            evidence_partition="balanced_distractor",
            partition_seed=17,
        )[0]

        alice_docs = first.partition_metadata["alice_documents"]
        bob_docs = first.partition_metadata["bob_documents"]
        all_docs = first.partition_metadata["single_reader_documents"]
        assert len(alice_docs) == len(bob_docs) == 5
        assert len(all_docs) == 10
        assert sum(doc["is_supporting"] for doc in alice_docs) == 1
        assert sum(doc["is_supporting"] for doc in bob_docs) == 1
        assert {doc["title"] for doc in alice_docs}.isdisjoint(
            {doc["title"] for doc in bob_docs}
        )
        assert {doc["title"] for doc in alice_docs + bob_docs} == {
            doc["title"] for doc in all_docs
        }
        assert first.evidence_alice == second.evidence_alice
        assert first.evidence_bob == second.evidence_bob
        assert first.evidence_all == second.evidence_all
        assert "is_supporting" not in first.evidence_alice
        assert "is_supporting" not in first.evidence_bob
        assert "is_supporting" not in first.evidence_all

    def test_balanced_distractor_rejects_non_ten_document_context(self):
        assert not is_valid_candidate(
            make_row("q-short"), "balanced_distractor"
        )


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

    def test_balanced_manifest_records_partition_policy(self, tmp_path: Path):
        path = tmp_path / "distractor_manifest.json"
        manifest = build_manifest(
            [make_distractor_row("q1")],
            1,
            selection_seed=0,
            manifest_path=path,
            evidence_partition="balanced_distractor",
            partition_seed=23,
        )
        assert manifest["evidence_partition"] == "balanced_distractor"
        assert manifest["partition_seed"] == 23
        loaded = load_manifest(path)[0]
        assert loaded.partition_metadata["distractors_included"] is True
        assert loaded.partition_metadata["gold_labels_exposed_to_models"] is False
