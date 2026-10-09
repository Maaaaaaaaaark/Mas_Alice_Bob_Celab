"""Fixed question selection and evidence partition.

Selects a configured number of HotpotQA validation questions and
deterministically partitions each question's evidence between Alice and Bob.

Selection rules (each rejected candidate is counted and recorded in the
manifest):

- ``answer`` is a non-empty string;
- ``type`` is ``bridge`` or ``comparison``;
- ``supporting_facts`` point to exactly 2 distinct document titles;
- both titles exist in ``context`` and have non-empty paragraphs.

The candidate list is shuffled with ``random.Random(sample_selection_seed)``
and the requested number of surviving candidates are kept.

Two versioned partition rules are supported. ``supporting_only`` preserves the
original experiment: supporting titles are sorted alphabetically and one full
paragraph goes to each worker. ``balanced_distractor`` requires the original
HotpotQA distractor context of ten distinct non-empty documents. It gives each
worker one supporting document and four distractors, then deterministically
shuffles the five documents within each worker. A separate deterministic
shuffle supplies all ten documents to the single-reader control. Gold labels
are retained only in manifest metadata and never rendered into model evidence.

The resulting manifest is saved as JSON and every run reads the same file,
so selection is identical across modes and re-runs.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .hotpotqa import (
    row_context_documents,
    row_document_paragraph,
    row_supporting_facts,
)


@dataclass
class SelectedQuestion:
    question_id: str  # HotpotQA "id" (legacy fixtures may use "_id")
    question: str
    answer: str
    q_type: str
    supporting_titles: List[str]  # 2 entries, alphabetical: [alice, bob]
    evidence_alice: str  # "Title: {title}\n{paragraph}"
    evidence_bob: str
    context_titles: List[str]  # titles of all context documents (record only)
    supporting_facts: List[List[Any]] = field(default_factory=list)
    hotpotqa_metadata: Dict[str, Any] = field(default_factory=dict)
    partition_metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_all: str = ""
    # Full ten-document pool in the ORIGINAL context order (pre-partition),
    # each entry {"title", "paragraph", "is_supporting"}. Populated only by
    # the ``balanced_distractor`` partition (training experiments); gold
    # labels live here for audit/trace and never enter rendered evidence.
    document_pool: List[Dict[str, Any]] = field(default_factory=list)


def _format_evidence(title: str, paragraph: str) -> str:
    return f"Title: {title}\n{paragraph}"


def _format_documents(documents: List[Dict[str, Any]]) -> str:
    """Render full documents without revealing supporting/distractor labels."""
    return "\n\n---\n\n".join(
        _format_evidence(str(doc["title"]), str(doc["paragraph"]))
        for doc in documents
    )


def _row_id(row: Dict[str, Any]) -> Optional[str]:
    """Best-effort stable row id for cross-split exclusion (None if absent)."""
    raw = row.get("id", row.get("_id"))
    return str(raw) if raw is not None else None


def _stable_rng(partition_seed: int, question_id: str, namespace: str) -> random.Random:
    """Return a stable per-question RNG independent of row and run order."""
    material = f"{partition_seed}:{question_id}:{namespace}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(derived)


def is_valid_candidate(
    row: Dict[str, Any], evidence_partition: str = "supporting_only"
) -> bool:
    """Check all hard selection rules for one HotpotQA row."""
    answer = row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return False
    if row.get("type") not in ("bridge", "comparison"):
        return False
    titles: List[str] = []
    try:
        supporting_facts = row_supporting_facts(row)
        context = row_context_documents(row)
    except (TypeError, ValueError):
        return False
    for pair in supporting_facts:
        if not isinstance(pair, (list, tuple)) or len(pair) < 1:
            return False
        title = pair[0]
        if not isinstance(title, str):
            return False
        if title not in titles:
            titles.append(title)
    if len(titles) != 2:
        return False
    context_titles = {doc.get("title") for doc in context}
    if not set(titles).issubset(context_titles):
        return False
    for doc in context:
        if doc.get("title") in titles:
            if not row_document_paragraph(row, doc["title"]).strip():
                return False
    if evidence_partition == "balanced_distractor":
        if len(context) != 10 or len(context_titles) != 10:
            return False
        if any(
            not isinstance(doc.get("title"), str)
            or not doc.get("title")
            or not " ".join(doc.get("sentences") or []).strip()
            for doc in context
        ):
            return False
        if len(context_titles - set(titles)) != 8:
            return False
    elif evidence_partition != "supporting_only":
        raise ValueError(f"unknown evidence partition: {evidence_partition}")
    return True


def _reject_reason(
    row: Dict[str, Any], evidence_partition: str = "supporting_only"
) -> Optional[str]:
    """Return a short reason string for a rejected candidate (for the manifest)."""
    answer = row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return "answer_missing_or_empty"
    if row.get("type") not in ("bridge", "comparison"):
        return "type_not_bridge_or_comparison"
    titles: List[str] = []
    try:
        supporting_facts = row_supporting_facts(row)
        context = row_context_documents(row)
    except (TypeError, ValueError):
        return "nested_fields_malformed"
    for pair in supporting_facts:
        title = pair[0] if isinstance(pair, (list, tuple)) and pair else None
        if not isinstance(title, str):
            return "supporting_facts_malformed"
        if title not in titles:
            titles.append(title)
    if len(titles) != 2:
        return "supporting_titles_count_not_2"
    context_titles = {doc.get("title") for doc in context}
    if not set(titles).issubset(context_titles):
        return "supporting_title_not_in_context"
    for doc in context:
        if doc.get("title") in titles:
            if not row_document_paragraph(row, doc["title"]).strip():
                return "supporting_paragraph_empty"
    if evidence_partition == "balanced_distractor":
        if len(context) != 10:
            return "context_document_count_not_10"
        if len(context_titles) != 10:
            return "context_titles_not_distinct"
        if any(
            not isinstance(doc.get("title"), str)
            or not doc.get("title")
            or not " ".join(doc.get("sentences") or []).strip()
            for doc in context
        ):
            return "context_document_missing_or_empty"
        if len(context_titles - set(titles)) != 8:
            return "distractor_count_not_8"
    return "supporting_paragraph_empty"


def select_questions(
    rows: List[Dict[str, Any]],
    num_questions: int,
    selection_seed: int,
    evidence_partition: str = "supporting_only",
    partition_seed: int = 0,
    exclude_ids: Optional[Set[str]] = None,
) -> List[SelectedQuestion]:
    """Select ``num_questions`` valid candidates after a seeded shuffle.

    ``exclude_ids`` (optional) skips rows whose id appears in the set; used
    by the training pipeline to keep val/test disjoint from train.
    """
    rng = random.Random(selection_seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    selected: List[SelectedQuestion] = []
    for index in order:
        if len(selected) >= num_questions:
            break
        row = rows[index]
        if exclude_ids is not None and _row_id(row) in exclude_ids:
            continue
        if not is_valid_candidate(row, evidence_partition):
            continue
        titles = []
        supporting_facts = row_supporting_facts(row)
        context_documents = row_context_documents(row)
        for pair in supporting_facts:
            if pair[0] not in titles:
                titles.append(pair[0])
        titles_sorted = sorted(titles)
        alice_title, bob_title = titles_sorted[0], titles_sorted[1]
        question_id = str(row.get("id", row.get("_id", index)))
        evidence_alice = _format_evidence(
            alice_title, row_document_paragraph(row, alice_title)
        )
        evidence_bob = _format_evidence(
            bob_title, row_document_paragraph(row, bob_title)
        )
        evidence_all = ""
        document_pool: List[Dict[str, Any]] = []
        partition_metadata: Dict[str, Any] = {
            "rule": "alphabetical_supporting_title",
            "alice_title": alice_title,
            "bob_title": bob_title,
            "distractors_included": False,
        }
        if evidence_partition == "balanced_distractor":
            documents = [
                {
                    "title": str(doc["title"]),
                    "paragraph": " ".join(doc.get("sentences") or []),
                    "is_supporting": doc["title"] in titles_sorted,
                }
                for doc in context_documents
            ]
            # Pre-partition pool: original context order, gold labels kept
            # for audit only (they never enter rendered evidence).
            document_pool = [dict(doc) for doc in documents]
            by_title = {doc["title"]: doc for doc in documents}
            distractors = [
                dict(doc) for doc in documents if not doc["is_supporting"]
            ]
            _stable_rng(partition_seed, question_id, "distractors").shuffle(
                distractors
            )
            alice_documents = [dict(by_title[alice_title]), *distractors[:4]]
            bob_documents = [dict(by_title[bob_title]), *distractors[4:]]
            _stable_rng(partition_seed, question_id, "alice_order").shuffle(
                alice_documents
            )
            _stable_rng(partition_seed, question_id, "bob_order").shuffle(
                bob_documents
            )
            all_documents = [dict(doc) for doc in documents]
            _stable_rng(partition_seed, question_id, "all_order").shuffle(
                all_documents
            )
            evidence_alice = _format_documents(alice_documents)
            evidence_bob = _format_documents(bob_documents)
            evidence_all = _format_documents(all_documents)
            partition_metadata = {
                "rule": "balanced_4_plus_4_distractor",
                "partition_seed": partition_seed,
                "distractors_included": True,
                "gold_labels_exposed_to_models": False,
                "alice_documents": [
                    {
                        "title": doc["title"],
                        "is_supporting": doc["is_supporting"],
                    }
                    for doc in alice_documents
                ],
                "bob_documents": [
                    {
                        "title": doc["title"],
                        "is_supporting": doc["is_supporting"],
                    }
                    for doc in bob_documents
                ],
                "single_reader_documents": [
                    {
                        "title": doc["title"],
                        "is_supporting": doc["is_supporting"],
                    }
                    for doc in all_documents
                ],
            }
        selected.append(
            SelectedQuestion(
                question_id=question_id,
                question=str(row["question"]),
                answer=str(row["answer"]),
                q_type=str(row["type"]),
                supporting_titles=list(titles_sorted),
                evidence_alice=evidence_alice,
                evidence_bob=evidence_bob,
                context_titles=[
                    doc.get("title", "") for doc in context_documents
                ],
                supporting_facts=[list(pair) for pair in supporting_facts],
                hotpotqa_metadata={
                    "level": row.get("level"),
                    "type": row.get("type"),
                },
                partition_metadata=partition_metadata,
                evidence_all=evidence_all,
                document_pool=document_pool,
            )
        )
    return selected


def build_manifest(
    rows: List[Dict[str, Any]],
    num_questions: int,
    selection_seed: int,
    manifest_path: Path,
    evidence_partition: str = "supporting_only",
    partition_seed: int = 0,
    exclude_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Select questions and write the manifest JSON file.

    The manifest records the selection rules, per-rule rejection counts,
    the selected questions (with partitions), and a sha256 of the
    ``questions`` payload for integrity checking on load.

    ``exclude_ids`` (optional) removes rows whose id appears in the set;
    excluded rows are counted under ``excluded_split_overlap``.
    """
    rejected_counts: Dict[str, int] = {}
    for row in rows:
        if exclude_ids is not None and _row_id(row) in exclude_ids:
            rejected_counts["excluded_split_overlap"] = (
                rejected_counts.get("excluded_split_overlap", 0) + 1
            )
            continue
        if is_valid_candidate(row, evidence_partition):
            continue
        reason = _reject_reason(row, evidence_partition) or "unknown"
        rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

    selected = select_questions(
        rows,
        num_questions,
        selection_seed,
        evidence_partition=evidence_partition,
        partition_seed=partition_seed,
        exclude_ids=exclude_ids,
    )
    if len(selected) < num_questions:
        raise RuntimeError(
            f"only {len(selected)} valid candidates found, need {num_questions}; "
            f"excluded_statistics={json.dumps(rejected_counts, sort_keys=True)}"
        )

    questions_payload = {
        "questions": [asdict(q) for q in selected],
    }
    payload_json = json.dumps(questions_payload, sort_keys=True, ensure_ascii=False)
    manifest = {
        "selection_seed": selection_seed,
        "num_questions_requested": num_questions,
        "num_questions_selected": len(selected),
        "selection_rules": [
            "answer is a non-empty string",
            "type in {bridge, comparison}",
            "supporting_facts point to exactly 2 distinct document titles",
            "both supporting titles exist in context",
            "both supporting paragraphs are non-empty",
        ],
        "evidence_partition": evidence_partition,
        "partition_seed": partition_seed,
        "partition_rule": (
            "supporting titles sorted alphabetically; first -> alice, second -> bob; "
            "no distractors are included"
            if evidence_partition == "supporting_only"
            else "one supporting document and four distractors per worker; "
            "worker and single-reader document order deterministically shuffled; "
            "gold labels retained only in manifest metadata"
        ),
        "excluded_statistics": rejected_counts,
        "total_rows_seen": len(rows),
        "questions_sha256": hashlib.sha256(
            payload_json.encode("utf-8")
        ).hexdigest(),
        "questions": [asdict(q) for q in selected],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def load_manifest(manifest_path: Path) -> List[SelectedQuestion]:
    """Load the manifest and verify its integrity, returning the questions."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    questions = [SelectedQuestion(**q) for q in manifest["questions"]]
    payload_json = json.dumps(
        {"questions": [asdict(q) for q in questions]},
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    if digest != manifest["questions_sha256"]:
        raise RuntimeError(
            f"manifest integrity check failed: {manifest_path} "
            "(questions payload hash mismatch)"
        )
    return questions
