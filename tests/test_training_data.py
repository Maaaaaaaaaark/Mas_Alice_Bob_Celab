"""Training data preparation: partition shape, isolation, integrity.

These tests are fully offline: the HuggingFace loader inside
``hotpot_mas.training.data`` is monkeypatched to return fixture rows, so
no dataset is downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import hotpot_mas.training.data as training_data
from hotpot_mas.question_selection import load_manifest
from hotpot_mas.training.config import TrainingConfig
from hotpot_mas.training.data import (
    assert_split_isolation,
    describe_split,
    load_splits,
    prepare_manifests,
)
from hotpot_mas.training.prompts_builder import TrainingPrompts
from tests.training_test_utils import (
    PROMPTS_TRAINING_DIR,
    build_balanced_manifest,
    make_distractor_row,
    make_rows,
    make_training_config,
    select_balanced_questions,
)


@pytest.fixture
def rows() -> List[Dict[str, Any]]:
    return make_rows(10)


@pytest.fixture
def prepared(
    tmp_path: Path, rows: List[Dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> Dict[str, Path]:
    monkeypatch.setattr(
        training_data,
        "load_hotpotqa_validation",
        lambda **kwargs: rows,
    )
    cfg = make_training_config(tmp_path)
    return prepare_manifests(cfg)


class TestPartition:
    def test_partition_is_reproducible(self, rows):
        first = select_balanced_questions(rows, 4, selection_seed=0)
        second = select_balanced_questions(rows, 4, selection_seed=0)
        assert [q.question_id for q in first] == [
            q.question_id for q in second
        ]
        assert [q.evidence_alice for q in first] == [
            q.evidence_alice for q in second
        ]
        assert [q.evidence_bob for q in first] == [
            q.evidence_bob for q in second
        ]

    def test_each_worker_gets_one_gold_four_distractors(self, rows):
        question = select_balanced_questions(rows, 1, selection_seed=0)[0]
        for key in ("alice_documents", "bob_documents"):
            docs = question.partition_metadata[key]
            assert len(docs) == 5
            supporting = [d for d in docs if d["is_supporting"]]
            assert len(supporting) == 1
            assert len({d["title"] for d in docs}) == 5
        # The 8 distractors are split 4/4 across the two workers.
        alice_titles = {
            d["title"]
            for d in question.partition_metadata["alice_documents"]
        }
        bob_titles = {
            d["title"]
            for d in question.partition_metadata["bob_documents"]
        }
        assert len(alice_titles & bob_titles) == 0
        assert len(alice_titles | bob_titles) == 10

    def test_document_pool_keeps_original_order_and_labels(self, rows):
        question = select_balanced_questions(rows, 1, selection_seed=0)[0]
        assert len(question.document_pool) == 10
        assert [d["title"] for d in question.document_pool] == [
            "Sup One",
            "Sup Two",
            *[f"Dis {i}" for i in range(8)],
        ]
        assert sum(d["is_supporting"] for d in question.document_pool) == 2

    def test_no_supporting_labels_in_model_inputs(self, rows):
        question = select_balanced_questions(rows, 1, selection_seed=0)[0]
        prompts = TrainingPrompts(PROMPTS_TRAINING_DIR)
        for worker in ("A", "B"):
            evidence = (
                question.evidence_alice
                if worker == "A"
                else question.evidence_bob
            )
            messages = prompts.worker_messages(
                worker, question.question, evidence
            )
            joined = " ".join(m["content"] for m in messages).lower()
            assert "supporting" not in joined
            assert "is_supporting" not in joined
            assert "distractor" not in joined


class TestSplitIsolation:
    def test_splits_pairwise_disjoint_and_counts(self, prepared):
        splits = load_splits(Path(prepared["train"]).parent)
        ids = {
            name: splits["splits"][name]["question_ids"]
            for name in ("train", "val", "test")
        }
        assert len(ids["train"]) == 4
        assert len(ids["val"]) == 2
        assert len(ids["test"]) == 2
        all_ids = ids["train"] + ids["val"] + ids["test"]
        assert len(all_ids) == len(set(all_ids))

    def test_exclusion_overlap_is_counted(self, tmp_path: Path, rows):
        manifest_path = tmp_path / "excluded.json"
        manifest = build_balanced_manifest(
            rows,
            3,
            selection_seed=0,
            manifest_path=manifest_path,
            exclude_ids={"q0", "q1"},
        )
        stats = manifest["excluded_statistics"]
        assert stats.get("excluded_split_overlap", 0) >= 2
        selected_ids = {q["question_id"] for q in manifest["questions"]}
        assert "q0" not in selected_ids
        assert "q1" not in selected_ids

    def test_assert_split_isolation_rejects_overlap(self):
        with pytest.raises(RuntimeError, match="disjoint"):
            assert_split_isolation(
                {"train": ["a", "b"], "val": ["b", "c"], "test": ["d"]}
            )

    def test_assert_split_isolation_rejects_dupes(self):
        with pytest.raises(RuntimeError, match="duplicate"):
            assert_split_isolation(
                {"train": ["a", "a"], "val": ["b"], "test": ["c"]}
            )

    def test_splits_integrity_check_fails_on_tamper(self, prepared):
        splits_path = Path(prepared["train"]).parent / "splits.json"
        payload = json.loads(splits_path.read_text(encoding="utf-8"))
        payload["splits"]["train"]["question_ids"].append("injected")
        splits_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError):
            load_splits(splits_path.parent)

    def test_manifest_integrity_via_load_manifest(self, prepared):
        questions = load_manifest(prepared["train"])
        assert len(questions) == 4
        assert all(q.document_pool for q in questions)

    def test_prepare_is_idempotent(self, prepared, rows, monkeypatch):
        # Re-running without --force reuses the existing splits.json and
        # returns the same manifest paths.
        tmp_path = Path(prepared["train"]).parent.parent
        monkeypatch.setattr(
            training_data,
            "load_hotpotqa_validation",
            lambda **kwargs: rows,
        )
        cfg = make_training_config(tmp_path)
        again = prepare_manifests(cfg)
        assert again == prepared
        manifests_dir = Path(prepared["train"]).parent
        assert (
            json.loads(
                (manifests_dir / "splits.json").read_text(encoding="utf-8")
            )["splits"]["train"]["question_ids"]
            == load_splits(manifests_dir)["splits"]["train"]["question_ids"]
        )

    def test_describe_split_reports_sources(self, tmp_path: Path):
        cfg = make_training_config(tmp_path)
        text = describe_split(cfg)
        assert "validation" in text
        assert "official test split has no public gold answers" in text
        assert "oracle-balanced partition" in text


class TestOneDistractorRow:
    def test_distractor_row_is_valid(self):
        row = make_distractor_row("q0")
        from hotpot_mas.question_selection import is_valid_candidate

        assert is_valid_candidate(row, "balanced_distractor")
