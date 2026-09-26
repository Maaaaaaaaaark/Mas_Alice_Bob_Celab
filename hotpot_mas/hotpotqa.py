"""HotpotQA dataset loading (spec sec. 5.1).

Loads the ``hotpot_qa`` dataset (config ``distractor``, split ``validation``)
through the HuggingFace ``datasets`` library, which mirrors the official
HotpotQA JSON fields used by the original evaluator. This module only loads
and normalizes field access; question filtering and evidence partitioning
live in ``question_selection.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List


def load_hotpotqa_validation(
    dataset_config: str = "distractor",
) -> List[Dict[str, Any]]:
    """Load HotpotQA validation rows as a list of raw dicts.

    ``trust_remote_code=True`` is required by the HotpotQA dataset script and
    only runs the HF hub script for this dataset.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "hotpot_qa",
        dataset_config,
        split="validation",
        trust_remote_code=True,
    )
    return [dict(row) for row in ds]


def row_supporting_facts(row: Dict[str, Any]) -> List[List]:
    """Return ``row["supporting_facts"]`` (list of [title, sent_id] pairs)."""
    return list(row.get("supporting_facts") or [])


def row_sentence(row: Dict[str, Any], title: str, sent_id: int) -> str:
    """Return one sentence of a context document by title and sent_id."""
    for doc in row.get("context") or []:
        if doc["title"] == title:
            return doc["sentences"][sent_id]
    raise KeyError(
        f"document {title!r} or sentence {sent_id} not found in context"
    )


def row_document_paragraph(row: Dict[str, Any], title: str) -> str:
    """Return a document's full text (all sentences joined) by title.

    Sentences are joined with spaces because the stored sentences are
    single-space separated pieces of the original paragraph.
    """
    for doc in row.get("context") or []:
        if doc["title"] == title:
            return " ".join(doc["sentences"])
    raise KeyError(f"document {title!r} not found in context")
