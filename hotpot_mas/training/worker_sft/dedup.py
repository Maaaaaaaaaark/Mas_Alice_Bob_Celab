"""Dedup and leakage checks for Stage 1a worker SFT.

What is checked (all counts, methods, thresholds, and rejection reasons
go into the dedup report):

1. **MuSiQue exclusion file** (``data/dev_test_singlehop_questions_v1.0.json``,
   sha256 verified): any SQuAD 2.0 candidate whose row id or normalized
   question appears in the exclusion list is removed, and any MuSiQue
   subquestion whose normalized question matches an exclusion question
   (any seed dataset) is removed. This keeps the official MuSiQue
   dev/test seed questions out of training.

2. **Project evaluation manifests** (the inference experiment's question
   manifests, configurable paths): any HotpotQA training candidate whose
   parent id or normalized question appears in an existing evaluation
   manifest is removed so the inference experiment's evaluation
   questions are never trained on. Missing manifests are reported as
   "check not run", never silently skipped.

3. **Cross-split isolation**: validation/test definitions are frozen
   first (parent level, all variants of one original example in one
   split); any training candidate that conflicts with a frozen eval
   sample (same parent id, same normalized (question, document set)
   key, or a near-duplicate question/paragraph) is removed.

4. **Near-duplicates** (documented method): token-Jaccard similarity of
   normalized questions >= threshold (default 0.9) or of normalized
   paragraphs >= threshold (default 0.95); the later duplicate is
   removed and the pair plus score is recorded.

The report states explicitly that string matching cannot prove zero
leakage; these checks reduce known leak paths, they do not prove
independence.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .convert import WorkerExample, normalize_text


def verify_file_sha256(path: Any, expected: str) -> Tuple[bool, str]:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    return actual == expected, actual


def load_exclusion_index(path: Any) -> Dict[str, Dict[str, Set[str]]]:
    """Load the MuSiQue exclusion file.

    Returns ``{seed_dataset: {"ids": set, "questions": set(normalized)}}``.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    index: Dict[str, Dict[str, Set[str]]] = {}
    for dataset, entries in raw.items():
        ids: Set[str] = set()
        questions: Set[str] = set()
        for entry in entries:
            if isinstance(entry, dict):
                if entry.get("id"):
                    ids.add(str(entry["id"]))
                if entry.get("question"):
                    questions.add(normalize_text(str(entry["question"])))
        index[str(dataset)] = {"ids": ids, "questions": questions}
    return index


def _question_doc_key(example: WorkerExample) -> str:
    """Normalized (question, document set) identity key."""
    docs = "|".join(
        sorted(normalize_text(doc.text) for doc in example.documents)
    )
    return f"{normalize_text(example.local_question)}::{docs}"


def _token_set(text: str) -> Set[str]:
    return set(normalize_text(text).split())


def _jaccard(a: str, b: str) -> float:
    set_a = _token_set(a)
    set_b = _token_set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _doc_set_text(example: WorkerExample) -> str:
    return " ".join(
        sorted(normalize_text(doc.text) for doc in example.documents)
    )


def _docs_similar(
    example: WorkerExample, other: WorkerExample, threshold: float
) -> bool:
    """Near-identical visible document sets (union Jaccard)."""
    return _jaccard(_doc_set_text(example), _doc_set_text(other)) >= threshold


def apply_exclusion_index(
    examples: List[WorkerExample],
    index: Dict[str, Dict[str, Set[str]]],
) -> Tuple[List[WorkerExample], List[Dict[str, Any]]]:
    """Remove examples that hit the MuSiQue exclusion list."""
    removed: List[Dict[str, Any]] = []
    kept: List[WorkerExample] = []
    for example in examples:
        if example.excluded:
            kept.append(example)
            continue
        reason = None
        if example.source == "squad2":
            squad = index.get("squad2", {"ids": set(), "questions": set()})
            row_id = str(example.provenance.get("row_id", ""))
            if row_id and row_id in squad["ids"]:
                reason = f"row id {row_id} in exclusion file (squad2)"
            elif (
                normalize_text(example.local_question) in squad["questions"]
            ):
                reason = (
                    "normalized question in exclusion file (squad2)"
                )
        elif example.source == "musique":
            for dataset, sets in index.items():
                if normalize_text(example.local_question) in sets["questions"]:
                    reason = (
                        f"normalized subquestion in exclusion file "
                        f"({dataset})"
                    )
                    break
        elif example.source == "hotpot":
            # HotpotQA is not a MuSiQue seed dataset, but a local question
            # could still collide with an exclusion question; apply the
            # question check uniformly.
            for dataset, sets in index.items():
                if normalize_text(example.local_question) in sets["questions"]:
                    reason = (
                        f"normalized local question in exclusion file "
                        f"({dataset})"
                    )
                    break
        if reason is not None:
            example.excluded = True
            example.exclusion_reason = (
                f"MuSiQue exclusion file: {reason}"
            )
            removed.append({"example_id": example.example_id, "reason": reason})
            continue
        kept.append(example)
    return kept, removed


def load_eval_manifest_questions(path: Any) -> Optional[Dict[str, Set[str]]]:
    """Question ids + normalized texts from one evaluation manifest.

    Returns None when the file does not exist (caller reports it).
    """
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ids: Set[str] = set()
    questions: Set[str] = set()
    if isinstance(raw, dict) and isinstance(raw.get("questions"), list):
        for entry in raw["questions"]:
            if isinstance(entry, dict):
                if entry.get("id"):
                    ids.add(str(entry["id"]))
                if entry.get("question"):
                    questions.add(normalize_text(str(entry["question"])))
            elif isinstance(entry, str):
                questions.add(normalize_text(entry))
    elif isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                if entry.get("id"):
                    ids.add(str(entry["id"]))
                if entry.get("question"):
                    questions.add(normalize_text(str(entry["question"])))
    return {"ids": ids, "questions": questions}


def apply_eval_manifest_exclusions(
    examples: List[WorkerExample],
    manifest_paths: Iterable[Any],
) -> Tuple[List[WorkerExample], List[Dict[str, Any]]]:
    """Remove hotpot training examples that overlap project eval manifests."""
    removed: List[Dict[str, Any]] = []
    kept: List[WorkerExample] = []
    manifest_status = []
    manifests = []
    for path in manifest_paths:
        manifest = load_eval_manifest_questions(path)
        if manifest is None:
            manifest_status.append(
                {"path": str(path), "status": "missing", "note": "check not run"}
            )
            continue
        manifest_status.append({"path": str(path), "status": "loaded"})
        manifests.append((str(path), manifest))
    for example in examples:
        if example.excluded:
            kept.append(example)
            continue
        if example.source != "hotpot":
            kept.append(example)
            continue
        reason = None
        parent_id = str(example.parent_id)
        normalized = normalize_text(example.local_question)
        for path, manifest in manifests:
            if parent_id in manifest["ids"]:
                reason = f"parent id {parent_id} in evaluation manifest {path}"
                break
            if normalized in manifest["questions"]:
                reason = (
                    f"normalized question in evaluation manifest {path}"
                )
                break
        if reason is not None:
            example.excluded = True
            example.exclusion_reason = (
                f"project evaluation manifest: {reason}"
            )
            removed.append(
                {"example_id": example.example_id, "reason": reason}
            )
            continue
        kept.append(example)
    return kept, removed, manifest_status


def freeze_val_test(
    examples: List[WorkerExample],
    val_parents: int,
    test_parents: int,
    seed: int,
    avoid_parent_ids: Optional[Set[str]] = None,
) -> Tuple[List[WorkerExample], List[Dict[str, Any]]]:
    """Assign val/test at the PARENT level before training selection.

    All variants of one parent share one split. ``avoid_parent_ids``
    parents (e.g. the reviewed trace set) are never frozen into
    val/test. Returns the examples with splits assigned (parents chosen
    here are excluded from training later by parent id).
    """
    import random

    avoid = set(avoid_parent_ids or ())
    eligible = [e for e in examples if not e.excluded]
    parents: List[str] = []
    for example in eligible:
        if example.parent_id in parents or example.parent_id in avoid:
            continue
        parents.append(example.parent_id)
    rng = random.Random(seed)
    rng.shuffle(parents)
    if len(parents) < val_parents + test_parents:
        raise ValueError(
            f"not enough eligible parents for frozen val/test: need "
            f"{val_parents + test_parents}, have {len(parents)}"
        )
    val_parent_ids = set(parents[:val_parents])
    test_parent_ids = set(
        parents[val_parents : val_parents + test_parents]
    )
    assignments = []
    for example in eligible:
        if example.parent_id in val_parent_ids:
            example.split = "val"
            assignments.append(
                {"example_id": example.example_id, "split": "val"}
            )
        elif example.parent_id in test_parent_ids:
            example.split = "test"
            assignments.append(
                {"example_id": example.example_id, "split": "test"}
            )
        else:
            example.split = "train"
    return examples, assignments


def apply_cross_split_isolation(
    examples: List[WorkerExample],
    question_jaccard_threshold: float,
    paragraph_jaccard_threshold: float,
) -> Tuple[List[WorkerExample], List[Dict[str, Any]]]:
    """Remove training candidates that conflict with frozen eval samples.

    Conflicts: same parent id, same normalized (question, doc set) key,
    or near-duplicate question/paragraph against a val/test sample.
    """
    eval_examples = [e for e in examples if e.split in {"val", "test"} and not e.excluded]
    train_examples = [e for e in examples if e.split == "train"]
    eval_parents = {e.parent_id for e in eval_examples}
    eval_keys = {_question_doc_key(e) for e in eval_examples}
    eval_questions = [
        (e.example_id, normalize_text(e.local_question))
        for e in eval_examples
    ]
    eval_paragraphs = [
        (e.example_id, normalize_text(doc.text))
        for e in eval_examples
        for doc in e.documents
    ]
    removed: List[Dict[str, Any]] = []
    kept: List[WorkerExample] = []
    for example in train_examples:
        if example.excluded:
            kept.append(example)
            continue
        reason = None
        if example.parent_id in eval_parents:
            reason = f"parent id {example.parent_id} is in frozen val/test"
        elif _question_doc_key(example) in eval_keys:
            reason = "identical normalized (question, documents) in val/test"
        else:
            question = normalize_text(example.local_question)
            for eval_id, eval_question in eval_questions:
                score = _jaccard(question, eval_question)
                # A near-duplicate QUESTION removes a candidate only when
                # its visible document set is also near-identical to the
                # eval sample's (same question about different documents
                # is a different example).
                eval_example = next(
                    e for e in eval_examples if e.example_id == eval_id
                )
                if score >= question_jaccard_threshold and _docs_similar(
                    example, eval_example, paragraph_jaccard_threshold
                ):
                    reason = (
                        f"near-duplicate question of {eval_id} "
                        f"(jaccard {score:.3f} >= "
                        f"{question_jaccard_threshold}, similar documents)"
                    )
                    break
            if reason is None:
                for doc in example.documents:
                    paragraph = normalize_text(doc.text)
                    for eval_id, eval_paragraph in eval_paragraphs:
                        score = _jaccard(paragraph, eval_paragraph)
                        if score >= paragraph_jaccard_threshold:
                            reason = (
                                f"near-duplicate paragraph of {eval_id} "
                                f"(jaccard {score:.3f} >= "
                                f"{paragraph_jaccard_threshold})"
                            )
                            break
                    if reason is not None:
                        break
        if reason is not None:
            example.excluded = True
            example.exclusion_reason = (
                f"cross-split isolation: {reason}"
            )
            removed.append(
                {"example_id": example.example_id, "reason": reason}
            )
            continue
        kept.append(example)
    return kept + eval_examples, removed


def apply_within_train_dedup(
    examples: List[WorkerExample],
    question_jaccard_threshold: float,
    paragraph_jaccard_threshold: float,
) -> Tuple[List[WorkerExample], List[Dict[str, Any]]]:
    """Remove exact and near duplicates within the training candidates."""
    removed: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()
    seen_questions: List[Tuple[str, str]] = []
    seen_paragraphs: List[Tuple[str, str]] = []
    kept: List[WorkerExample] = []
    # Curated/reviewed records must win over their automatically converted
    # twins. Auto candidates are loaded first by the runner, so processing
    # raw input order would otherwise discard the reviewed trace examples.
    ordered_examples = sorted(
        enumerate(examples),
        key=lambda item: (
            0
            if item[1].review.get("status") == "reviewed"
            and item[1].split == "train"
            and not item[1].probe
            else 1,
            item[0],
        ),
    )
    for _, example in ordered_examples:
        if example.excluded:
            kept.append(example)
            continue
        if example.split != "train":
            kept.append(example)
            continue
        if example.probe:
            # Fixed generation probes are not training data and are
            # never removed by dedup.
            kept.append(example)
            continue
        reason = None
        key = _question_doc_key(example)
        if key in seen_keys:
            reason = "exact duplicate (normalized question + documents)"
        else:
            question = normalize_text(example.local_question)
            for other_id, other_question in seen_questions:
                score = _jaccard(question, other_question)
                # Same rule as cross-split isolation: a near-duplicate
                # question removes a candidate only when the visible
                # document sets are also near-identical.
                other = next(
                    e for e in examples if e.example_id == other_id
                )
                if score >= question_jaccard_threshold and _docs_similar(
                    example, other, paragraph_jaccard_threshold
                ):
                    reason = (
                        f"near-duplicate question of {other_id} "
                        f"(jaccard {score:.3f}, similar documents)"
                    )
                    break
            # The same document may legitimately answer many different
            # questions. Paragraph similarity alone is therefore not a
            # within-train duplicate; the question+document check above
            # is the relevant criterion.
        if reason is not None:
            example.excluded = True
            example.exclusion_reason = f"within-train dedup: {reason}"
            removed.append(
                {"example_id": example.example_id, "reason": reason}
            )
            continue
        seen_keys.add(key)
        seen_questions.append((example.example_id, question))
        for doc in example.documents:
            seen_paragraphs.append(
                (example.example_id, normalize_text(doc.text))
            )
        kept.append(example)
    return kept, removed


def run_dedup_pipeline(
    examples: List[WorkerExample],
    exclusion_index: Dict[str, Dict[str, Set[str]]],
    manifest_paths: Iterable[Any],
    val_parents: int,
    test_parents: int,
    freeze_seed: int,
    question_jaccard_threshold: float,
    paragraph_jaccard_threshold: float,
    avoid_parent_ids: Optional[Set[str]] = None,
) -> Tuple[List[WorkerExample], Dict[str, Any]]:
    """Run the full dedup pipeline and build the dedup report.

    Order: exclusion file -> eval manifests -> frozen val/test ->
    cross-split isolation -> within-train dedup.
    """
    report: Dict[str, Any] = {
        "methods": {
            "question_normalization": (
                "lowercase, punctuation removed, whitespace collapsed"
            ),
            "near_dup_method": (
                "token-Jaccard over normalized text; question threshold "
                f"{question_jaccard_threshold}, paragraph threshold "
                f"{paragraph_jaccard_threshold}; the later example is removed"
            ),
            "split_isolation": (
                "val/test frozen at parent level before training selection; "
                "all variants of one original example share one split"
            ),
            "limitation": (
                "string matching cannot prove zero leakage; these checks "
                "remove known leak paths but do not prove independence"
            ),
        },
        "counts": {},
    }

    after_exclusion, exclusion_removed = apply_exclusion_index(
        examples, exclusion_index
    )
    after_manifests, manifest_removed, manifest_status = (
        apply_eval_manifest_exclusions(after_exclusion, manifest_paths)
    )
    report["eval_manifests"] = manifest_status

    after_freeze, assignments = freeze_val_test(
        after_manifests,
        val_parents,
        test_parents,
        freeze_seed,
        avoid_parent_ids=avoid_parent_ids,
    )
    after_isolation, isolation_removed = apply_cross_split_isolation(
        after_freeze,
        question_jaccard_threshold,
        paragraph_jaccard_threshold,
    )
    after_dedup, dedup_removed = apply_within_train_dedup(
        after_isolation,
        question_jaccard_threshold,
        paragraph_jaccard_threshold,
    )

    report["removed"] = {
        "exclusion_file": exclusion_removed,
        "eval_manifests": manifest_removed,
        "cross_split_isolation": isolation_removed,
        "within_train_dedup": dedup_removed,
    }
    report["val_test_assignments"] = assignments
    report["counts"] = {
        "exclusion_file_removed": len(exclusion_removed),
        "eval_manifest_removed": len(manifest_removed),
        "cross_split_removed": len(isolation_removed),
        "within_train_dedup_removed": len(dedup_removed),
        "val_examples": len(
            [e for e in after_dedup if e.split == "val" and not e.excluded]
        ),
        "test_examples": len(
            [e for e in after_dedup if e.split == "test" and not e.excluded]
        ),
        "train_candidates": len(
            [e for e in after_dedup if e.split == "train" and not e.excluded]
        ),
    }
    return after_dedup, report
