"""Dedup / split-isolation / leakage checks for Stage 1a."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from hotpot_mas.training.worker_sft.dedup import (
    apply_cross_split_isolation,
    apply_eval_manifest_exclusions,
    apply_exclusion_index,
    apply_within_train_dedup,
    freeze_val_test,
    load_eval_manifest_questions,
    load_exclusion_index,
    run_dedup_pipeline,
    verify_file_sha256,
)

from worker_sft_test_utils import make_doc, make_evidence, make_example


def _example(example_id, question, text, source="squad2", **kw: Any) -> Any:
    doc = make_doc("d0", text)
    return make_example(
        example_id,
        question,
        [doc],
        source=source,
        evidence=(
            [make_evidence("d0", text)] if kw.pop("answerable", True)
            else []
        ),
        **kw,
    )


# --------------------------------------------------------------------------
# Exclusion file (MuSiQue official leak-prevention list)
# --------------------------------------------------------------------------

def test_verify_file_sha256(tmp_path):
    path = tmp_path / "f.json"
    path.write_text("hello", encoding="utf-8")
    assert verify_file_sha256(path, "x") == (
        False,
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
    )
    assert verify_file_sha256(path, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")[0]


def _exclusion_index() -> Dict[str, Dict[str, Any]]:
    return {
        "squad2": {
            "ids": {"leak-id-1"},
            "questions": {"who leaked this question"},
        },
        "hotpotqa": {"ids": set(), "questions": {"how old is the bridge"}},
    }


def test_exclusion_file_removes_squad2_by_row_id_and_question():
    leak_id = _example("s1", "Some question?", "Text.")
    leak_id.provenance["row_id"] = "leak-id-1"
    leak_q = _example("s2", "Who leaked this question?", "Text.")
    clean = _example("s3", "A clean question?", "Text.")
    kept, removed = apply_exclusion_index(
        [leak_id, leak_q, clean], _exclusion_index()
    )
    assert [e.example_id for e in kept] == ["s3"]
    assert {r["example_id"] for r in removed} == {"s1", "s2"}
    assert leak_id.exclusion_reason.startswith("MuSiQue exclusion file")


def test_exclusion_file_removes_musique_by_normalized_question():
    musique = _example(
        "m1", "How old is the bridge", "Text.", source="musique"
    )
    kept, removed = apply_exclusion_index([musique], _exclusion_index())
    assert kept == []
    assert removed[0]["example_id"] == "m1"


def test_exclusion_file_skips_already_excluded_examples():
    example = _example("s1", "Who leaked this question?", "Text.")
    example.excluded = True
    example.exclusion_reason = "earlier"
    kept, removed = apply_exclusion_index([example], _exclusion_index())
    assert kept == [example] and removed == []


# --------------------------------------------------------------------------
# Project evaluation manifests (hotpot candidates only)
# --------------------------------------------------------------------------

def test_eval_manifest_removes_hotpot_by_parent_id_and_question(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {"id": "hotpot-parent-9", "question": "Question A?"},
                    {"id": "other", "question": "Question B?"},
                ]
            }
        ),
        encoding="utf-8",
    )
    by_parent = _example(
        "h1", "Local question.", "Text.", source="hotpot",
        parent_id="hotpot-parent-9",
    )
    by_question = _example(
        "h2", "Question B?", "Text.", source="hotpot",
        parent_id="other-parent",
    )
    untouched = _example(
        "h3", "Safe question.", "Text.", source="hotpot",
        parent_id="safe-parent",
    )
    kept, removed, status = apply_eval_manifest_exclusions(
        [by_parent, by_question, untouched], [path]
    )
    assert [e.example_id for e in kept] == ["h3"]
    assert {r["example_id"] for r in removed} == {"h1", "h2"}
    assert status == [{"path": str(path), "status": "loaded"}]


def test_missing_eval_manifest_is_reported_not_skipped(tmp_path):
    kept, removed, status = apply_eval_manifest_exclusions(
        [], [tmp_path / "missing.json"]
    )
    assert kept == [] and removed == []
    assert status[0]["status"] == "missing"
    assert status[0]["note"] == "check not run"


def test_eval_manifests_only_apply_to_hotpot():
    squad = _example("s1", "Question B?", "Text.", source="squad2")
    kept, _, _ = apply_eval_manifest_exclusions(
        [squad], []
    )
    assert kept == [squad]


# --------------------------------------------------------------------------
# Frozen val/test at parent level
# --------------------------------------------------------------------------

def _parents(parent_id: str, count: int, source="squad2") -> List[Any]:
    return [
        _example(
            f"{parent_id}-v{i}",
            f"Question {parent_id} variant {i}?",
            f"Text for {parent_id} variant {i}.",
            source=source,
            parent_id=parent_id,
        )
        for i in range(count)
    ]


def test_freeze_val_test_is_parent_level_and_deterministic():
    examples = _parents("p1", 2) + _parents("p2", 2) + _parents("p3", 2)
    first, _ = freeze_val_test(
        [e for e in examples], val_parents=1, test_parents=1, seed=7
    )
    second, _ = freeze_val_test(
        [e for e in examples], val_parents=1, test_parents=1, seed=7
    )
    assert [e.split for e in first] == [e.split for e in second]
    # All variants of one parent share one split.
    by_parent: Dict[str, Any] = {}
    for e in first:
        by_parent.setdefault(e.parent_id, set()).add(e.split)
    assert all(len(splits) == 1 for splits in by_parent.values())
    # Exactly one val parent and one test parent.
    val_parents = {p for p, s in by_parent.items() if s == {"val"}}
    test_parents = {p for p, s in by_parent.items() if s == {"test"}}
    assert len(val_parents) == 1 and len(test_parents) == 1
    assert val_parents != test_parents


def test_freeze_val_test_respects_avoid_parent_ids():
    examples = _parents("reviewed", 2) + _parents("a", 2) + _parents("b", 2)
    freeze_val_test(
        examples, val_parents=1, test_parents=1, seed=7,
        avoid_parent_ids={"reviewed"},
    )
    reviewed = [e for e in examples if e.parent_id == "reviewed"]
    assert all(e.split == "train" for e in reviewed)


def test_freeze_val_test_fails_loudly_when_short():
    examples = _parents("p1", 2)
    with pytest.raises(ValueError, match="not enough eligible parents"):
        freeze_val_test(examples, val_parents=1, test_parents=1, seed=7)


# --------------------------------------------------------------------------
# Cross-split isolation
# --------------------------------------------------------------------------

def test_cross_split_isolation_removes_conflicting_parents_and_paragraphs():
    train = _example(
        "t1", "A training question?", "Shared paragraph text here.",
        parent_id="eval-parent",
    )
    eval_example = _example(
        "e1", "An eval question?", "Shared paragraph text here.",
        parent_id="eval-parent",
    )
    eval_example.split = "val"
    kept, removed = apply_cross_split_isolation(
        [train, eval_example], 0.9, 0.95
    )
    assert train.excluded
    assert any("parent id" in r["reason"] for r in removed)


def test_cross_split_isolation_near_dup_question_needs_similar_docs():
    # Same question asked about DIFFERENT documents must survive.
    train = _example(
        "t2",
        "What color is the rare mineral?",
        "Doc A about totally different content with many words here.",
    )
    eval_example = _example(
        "e2",
        "What color is the rare mineral?",
        "Doc B about completely other content with many words here.",
    )
    eval_example.split = "val"
    kept, removed = apply_cross_split_isolation(
        [train, eval_example], 0.9, 0.95
    )
    assert not train.excluded
    assert removed == []


def test_cross_split_isolation_removes_near_dup_question_with_same_docs():
    # Jaccard("what color is the rare mineral anyway",
    #        "what color is the rare mineral") = 7/8 = 0.875.
    train = _example(
        "t3",
        "What color is the rare mineral anyway?",
        "The rare mineral is azure blue and shiny in daylight.",
    )
    eval_example = _example(
        "e3",
        "What color is the rare mineral?",
        "The rare mineral is azure blue and shiny in daylight.",
    )
    eval_example.split = "val"
    kept, removed = apply_cross_split_isolation(
        [train, eval_example], 0.8, 0.95
    )
    assert train.excluded
    assert any("near-duplicate" in r["reason"] for r in removed)


def test_cross_split_isolation_removes_near_duplicate_paragraph():
    train = _example(
        "t4",
        "Unrelated question about dogs?",
        "The blue mineral glows under ultraviolet light at night.",
    )
    eval_example = _example(
        "e4",
        "Unrelated question about cats?",
        "The blue mineral glows under ultraviolet light at night.",
    )
    eval_example.split = "val"
    kept, removed = apply_cross_split_isolation(
        [train, eval_example], 0.9, 0.95
    )
    assert train.excluded
    assert any("near-duplicate paragraph" in r["reason"] for r in removed)


# --------------------------------------------------------------------------
# Within-train dedup
# --------------------------------------------------------------------------

def test_within_train_dedup_removes_exact_duplicates_only():
    first = _example("a1", "Who built the bridge?", "The bridge is old.")
    dup = _example("a2", "Who built the bridge?", "The bridge is old.")
    kept, removed = apply_within_train_dedup([first, dup], 0.9, 0.95)
    assert [e.example_id for e in kept] == ["a1"]
    assert removed[0]["example_id"] == "a2"


def test_within_train_dedup_prefers_reviewed_duplicate():
    automatic = _example(
        "auto", "Who built the bridge?", "The bridge is old."
    )
    automatic.review = {"status": "auto_checked"}
    reviewed = _example(
        "reviewed", "Who built the bridge?", "The bridge is old."
    )
    reviewed.review = {"status": "reviewed"}
    kept, removed = apply_within_train_dedup(
        [automatic, reviewed], 0.9, 0.95
    )
    assert [e.example_id for e in kept] == ["reviewed"]
    assert [r["example_id"] for r in removed] == ["auto"]


def test_within_train_dedup_exempts_probes():
    first = _example("a1", "Who built the bridge?", "The bridge is old.")
    probe = _example(
        "probe-1", "Who built the bridge?", "The bridge is old.",
        probe=True,
    )
    kept, removed = apply_within_train_dedup([first, probe], 0.9, 0.95)
    assert {e.example_id for e in kept} == {"a1", "probe-1"}
    assert removed == []


def test_within_train_dedup_keeps_same_question_with_different_docs():
    first = _example(
        "a1",
        "What color is the mineral?",
        "Mineral one is azure blue and shiny in the daylight here.",
    )
    other = _example(
        "a2",
        "What color is the mineral?",
        "Mineral two is crimson red and dull in the dark cave there.",
    )
    kept, removed = apply_within_train_dedup([first, other], 0.9, 0.95)
    assert {e.example_id for e in kept} == {"a1", "a2"}
    assert removed == []


def test_within_train_dedup_keeps_same_paragraph_for_different_questions():
    first = _example(
        "a1", "Question about dogs?", "The blue mineral glows at night."
    )
    other = _example(
        "a2", "Question about cats?", "The blue mineral glows at night."
    )
    kept, removed = apply_within_train_dedup([first, other], 0.9, 0.95)
    assert [e.example_id for e in kept] == ["a1", "a2"]
    assert removed == []


# --------------------------------------------------------------------------
# Full pipeline + report
# --------------------------------------------------------------------------

def test_run_dedup_pipeline_builds_report_with_counts_and_limitation():
    examples = _parents("p1", 2) + _parents("p2", 2) + _parents("p3", 2)
    after, report = run_dedup_pipeline(
        examples,
        _exclusion_index(),
        [],
        val_parents=1,
        test_parents=1,
        freeze_seed=7,
        question_jaccard_threshold=0.9,
        paragraph_jaccard_threshold=0.95,
    )
    assert "limitation" in report["methods"]
    assert "cannot prove zero leakage" in report["methods"]["limitation"]
    assert report["counts"]["val_examples"] >= 1
    assert report["counts"]["test_examples"] >= 1
    assert report["counts"]["train_candidates"] >= 1
    assert "removed" in report


def test_load_exclusion_index_builds_ids_and_questions(tmp_path):
    path = tmp_path / "excl.json"
    path.write_text(
        json.dumps(
            {
                "squad2": [{"id": "x1", "question": "A question?"}],
                "hotpotqa": [{"id": "y1", "question": "B question?"}],
            }
        ),
        encoding="utf-8",
    )
    index = load_exclusion_index(path)
    assert index["squad2"]["ids"] == {"x1"}
    assert "a question" in index["squad2"]["questions"]
    assert index["hotpotqa"]["ids"] == {"y1"}


def test_load_eval_manifest_questions_returns_none_when_missing(tmp_path):
    assert (
        load_eval_manifest_questions(tmp_path / "absent.json") is None
    )
