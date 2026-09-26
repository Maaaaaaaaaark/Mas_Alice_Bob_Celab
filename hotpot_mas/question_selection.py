"""Fixed question selection and evidence partition (spec sec. 5.2).

Selects exactly 100 HotpotQA validation questions and deterministically
partitions each question's two supporting documents between Alice and Bob.

Selection rules (each rejected candidate is counted and recorded in the
manifest):

- ``answer`` is a non-empty string;
- ``type`` is ``bridge`` or ``comparison``;
- ``supporting_facts`` point to exactly 2 distinct document titles;
- both titles exist in ``context`` and have non-empty paragraphs.

The candidate list is shuffled with ``random.Random(sample_selection_seed)``
and the first 100 surviving candidates are kept.

Partition rule (deterministic, no RNG): the two supporting titles are sorted
alphabetically; the first goes to Alice, the second to Bob. Each worker
receives the full paragraph of its supporting document, formatted
``Title: {title}\n{paragraph}``. Distractors are never included (spec sec. 5).

The resulting manifest is saved as JSON and every run reads the same file,
so selection is identical across modes and re-runs.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _format_evidence(title: str, paragraph: str) -> str:
    return f"Title: {title}\n{paragraph}"


def is_valid_candidate(row: Dict[str, Any]) -> bool:
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
    return True


def _reject_reason(row: Dict[str, Any]) -> Optional[str]:
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
    return "supporting_paragraph_empty"


def select_questions(
    rows: List[Dict[str, Any]],
    num_questions: int,
    selection_seed: int,
) -> List[SelectedQuestion]:
    """Select ``num_questions`` valid candidates after a seeded shuffle."""
    rng = random.Random(selection_seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    selected: List[SelectedQuestion] = []
    for index in order:
        if len(selected) >= num_questions:
            break
        row = rows[index]
        if not is_valid_candidate(row):
            continue
        titles = []
        supporting_facts = row_supporting_facts(row)
        context_documents = row_context_documents(row)
        for pair in supporting_facts:
            if pair[0] not in titles:
                titles.append(pair[0])
        titles_sorted = sorted(titles)
        alice_title, bob_title = titles_sorted[0], titles_sorted[1]
        selected.append(
            SelectedQuestion(
                question_id=str(row.get("id", row.get("_id", index))),
                question=str(row["question"]),
                answer=str(row["answer"]),
                q_type=str(row["type"]),
                supporting_titles=list(titles_sorted),
                evidence_alice=_format_evidence(
                    alice_title, row_document_paragraph(row, alice_title)
                ),
                evidence_bob=_format_evidence(
                    bob_title, row_document_paragraph(row, bob_title)
                ),
                context_titles=[
                    doc.get("title", "") for doc in context_documents
                ],
                supporting_facts=[list(pair) for pair in supporting_facts],
                hotpotqa_metadata={
                    "level": row.get("level"),
                    "type": row.get("type"),
                },
                partition_metadata={
                    "rule": "alphabetical_supporting_title",
                    "alice_title": alice_title,
                    "bob_title": bob_title,
                    "distractors_included": False,
                },
            )
        )
    return selected


def build_manifest(
    rows: List[Dict[str, Any]],
    num_questions: int,
    selection_seed: int,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Select questions and write the manifest JSON file.

    The manifest records the selection rules, per-rule rejection counts,
    the selected questions (with partitions), and a sha256 of the
    ``questions`` payload for integrity checking on load.
    """
    rejected_counts: Dict[str, int] = {}
    for row in rows:
        if is_valid_candidate(row):
            continue
        reason = _reject_reason(row) or "unknown"
        rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

    selected = select_questions(rows, num_questions, selection_seed)
    if len(selected) < num_questions:
        raise RuntimeError(
            f"only {len(selected)} valid candidates found, need {num_questions}"
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
        "partition_rule": (
            "supporting titles sorted alphabetically; first -> alice, second -> bob; "
            "no distractors are included"
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
