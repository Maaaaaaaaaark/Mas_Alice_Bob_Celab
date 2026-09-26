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
    dataset_name: str = "hotpot_qa",
    split: str = "validation",
) -> List[Dict[str, Any]]:
    """Load HotpotQA validation rows as a list of raw dicts.

    The Hub dataset is Parquet-backed, so no remote dataset code is needed.
    """
    from datasets import load_dataset

    ds = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
    )
    return [dict(row) for row in ds]


def row_supporting_facts(row: Dict[str, Any]) -> List[List]:
    """Return supporting facts as ``[title, sent_id]`` pairs.

    Hugging Face currently exposes this Sequence feature as a dict of
    parallel lists.  Older/local JSON fixtures commonly use a list of pairs,
    so both representations are accepted at this boundary.
    """
    facts = row.get("supporting_facts") or []
    if isinstance(facts, dict):
        titles = facts.get("title") or []
        sent_ids = facts.get("sent_id") or []
        if len(titles) != len(sent_ids):
            raise ValueError("supporting_facts title/sent_id lengths differ")
        return [[title, sent_id] for title, sent_id in zip(titles, sent_ids)]
    return [list(pair) for pair in facts]


def row_context_documents(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return context as a list of ``{title, sentences}`` dictionaries."""
    context = row.get("context") or []
    if isinstance(context, dict):
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        if len(titles) != len(sentences):
            raise ValueError("context title/sentences lengths differ")
        return [
            {"title": title, "sentences": doc_sentences}
            for title, doc_sentences in zip(titles, sentences)
        ]
    return [dict(doc) for doc in context]


def row_sentence(row: Dict[str, Any], title: str, sent_id: int) -> str:
    """Return one sentence of a context document by title and sent_id."""
    for doc in row_context_documents(row):
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
    for doc in row_context_documents(row):
        if doc["title"] == title:
            return " ".join(doc["sentences"])
    raise KeyError(f"document {title!r} not found in context")
