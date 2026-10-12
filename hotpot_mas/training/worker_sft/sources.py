"""Source loaders for Stage 1a worker SFT.

Each source loader returns ``(rows, provenance)`` where provenance
records the dataset identity, revision/version when available, and how
it was loaded. Missing or failed sources are reported explicitly — a
source that is enabled but fails to load is a hard error, and the
optional teacher file's absence is reported in the data manifest, never
silently omitted.

The MuSiQue official JSONL (``musique_ans_v1.0_train.jsonl``) is loaded
through the Hub mirror ``dgslibisey/MuSiQue``, which stores the official
fields; the original distribution is Google Drive (see
``docs/WORKER_SFT.md``). SQuAD 2.0 is ``squad_v2``; HotpotQA train is
``hotpot_qa`` / ``distractor`` (same loader as the inference baseline,
which stays untouched).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from hotpot_mas.hotpotqa import load_hotpotqa_validation, row_context_documents


def load_squad2(split: str = "train") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("squad_v2", split=split)
    provenance = {
        "dataset": "squad_v2",
        "split": split,
        "version": (
            str(ds._info.version) if getattr(ds, "_info", None) else None
        ),
        "builder": (
            ds._info.builder_name if getattr(ds, "_info", None) else None
        ),
        "note": "official SQuAD 2.0; unanswerable rows keep their official label",
    }
    return [dict(row) for row in ds], provenance


def load_musique(split: str = "train") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load the official MuSiQue ans v1.0 JSONL via its Hub mirror."""
    from datasets import load_dataset

    ds = load_dataset("dgslibisey/MuSiQue", split=split)
    provenance = {
        "dataset": "musique_ans_v1.0",
        "hub_mirror": "dgslibisey/MuSiQue",
        "split": split,
        "version": (
            str(ds._info.version) if getattr(ds, "_info", None) else None
        ),
        "note": (
            "official MuSiQue ans v1.0 JSONL mirrored on the Hub; the "
            "original distribution is Google Drive (see docs/WORKER_SFT.md)"
        ),
    }
    return [dict(row) for row in ds], provenance


def load_hotpotqa_train(
    dataset_config: str = "distractor",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = load_hotpotqa_validation(
        dataset_config=dataset_config,
        dataset_name="hotpot_qa",
        split="train",
    )
    provenance = {
        "dataset": "hotpot_qa",
        "config": dataset_config,
        "split": "train",
        "note": (
            "official HotpotQA train rows; training examples come only "
            "from the reviewed (and optional teacher) files"
        ),
    }
    return rows, provenance


def index_rows_by_id(
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """``id -> row`` index for reviewed-entry resolution."""
    return {str(row["id"]): row for row in rows}


def build_reviewed_loaders(
    squad2_rows: List[Dict[str, Any]],
    musique_rows: List[Dict[str, Any]],
    hotpot_rows: List[Dict[str, Any]],
) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """Row resolvers for reviewed entries, keyed by source."""
    squad_index = index_rows_by_id(squad2_rows)
    musique_index = index_rows_by_id(musique_rows)
    hotpot_index = index_rows_by_id(hotpot_rows)

    def load_squad2_row(pointer: Dict[str, Any]) -> Dict[str, Any]:
        return squad_index[str(pointer["row_id"])]

    def load_musique_row(pointer: Dict[str, Any]) -> Dict[str, Any]:
        return musique_index[str(pointer["row_id"])]

    def load_hotpot_row(pointer: Dict[str, Any]) -> Dict[str, Any]:
        return hotpot_index[str(pointer["row_id"])]

    return {
        "squad2": load_squad2_row,
        "musique": load_musique_row,
        "hotpot": load_hotpot_row,
    }


def hotpot_row_documents(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Context documents as ``{title, sentences}`` for one hotpot row."""
    return row_context_documents(row)
