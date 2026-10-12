"""Shared builders for the Stage 1a worker-SFT test suite (not collected
by pytest: the filename does not match the ``test_*.py`` pattern).

Everything runs offline on CPU: fixture example records, a fake
tokenizer/model pair (``hotpot_mas.training.worker_sft.fakes``), and
``WorkerSFTConfig`` instances pointing at temporary directories plus the
real ``prompts_worker_sft/`` instruction file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from hotpot_mas.training.worker_sft.config import (
    WorkerSFTConfig,
    WorkerSFTDecodeConfig,
    WorkerSFTMixConfig,
    WorkerSFTModelConfig,
    WorkerSFTReviewConfig,
    WorkerSFTTrainConfig,
)
from hotpot_mas.training.worker_sft.convert import (
    EvidenceSentence,
    WorkerDocument,
    WorkerExample,
)
from hotpot_mas.training.worker_sft.fakes import build_fake_model

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_WORKER_SFT_DIR = REPO_ROOT / "prompts_worker_sft"

QUESTION_WORDS = ["Question:", "Documents:", "NOT", "FOUND"]


def make_doc(
    doc_id: str,
    text: str,
    title: str = "",
) -> WorkerDocument:
    from hotpot_mas.training.worker_sft.convert import normalize_text

    doc = WorkerDocument(doc_id=doc_id, title=title, text=text)
    doc.normalized_text = normalize_text(text)
    return doc


def make_example(
    example_id: str,
    question: str,
    docs: List[WorkerDocument],
    *,
    answer: str = "",
    aliases: Optional[List[str]] = None,
    evidence: Optional[List[EvidenceSentence]] = None,
    source: str = "squad2",
    parent_id: Optional[str] = None,
    answerable: Optional[bool] = None,
    negative_reason: Optional[str] = None,
    split: str = "train",
    probe: bool = False,
    review_status: str = "reviewed",
) -> WorkerExample:
    if answerable is None:
        answerable = bool(answer) or bool(evidence)
    evidence = list(evidence or [])
    example = WorkerExample(
        example_id=example_id,
        source=source,
        split=split,
        parent_id=parent_id or example_id,
        original_question=question,
        local_question=question,
        substitutions=[],
        answer=answer,
        answer_aliases=list(aliases or []),
        answerable=answerable,
        documents=docs,
        evidence=evidence,
        negative_reason=negative_reason if not answerable else None,
        probe=probe,
        review={
            "status": review_status,
            "reviewer": "test",
            "method": "test fixture",
            "notes": "",
        },
    )
    example.provenance["row_id"] = example_id
    for ev in evidence:
        doc = next(d for d in docs if d.doc_id == ev.doc_id)
        start = doc.text.find(ev.sentence)
        if start < 0:
            ev.char_span = None
        else:
            ev.char_span = (start, start + len(ev.sentence))
    return example


def make_evidence(
    doc_id: str,
    sentence: str,
    sentence_index: int = 0,
) -> EvidenceSentence:
    return EvidenceSentence(
        doc_id=doc_id,
        sentence_index=sentence_index,
        sentence=sentence,
        char_span=None,
    )


def corpus_for(examples: List[WorkerExample]) -> List[str]:
    """Every string the fake tokenizer must know for these examples."""
    texts = QUESTION_WORDS + [
        (PROMPTS_WORKER_SFT_DIR / "worker_system.txt").read_text(
            encoding="utf-8"
        )
    ]
    for example in examples:
        texts.append(example.local_question)
        texts.append(example.answer)
        texts.extend(example.answer_aliases)
        for doc in example.documents:
            texts.append(doc.title)
            texts.append(doc.text)
        for ev in example.evidence:
            texts.append(ev.sentence)
    return texts


def build_cpu_config(
    tmp_path: Path,
    *,
    version: str = "v0",
    grad_accum_steps: int = 2,
    num_epochs: int = 1,
    require_reviewed: bool = False,
    total_examples: int = 0,
    max_input_tokens: int = 8192,
) -> WorkerSFTConfig:
    """A CPU config pointing at temporary dirs and the real prompt file."""
    return WorkerSFTConfig(
        experiment_id="test_worker_sft",
        experiment_version=version,
        seed=0,
        output_dir=tmp_path / "outputs",
        data_dir=tmp_path / "worker_sft_data",
        prompt_dir=PROMPTS_WORKER_SFT_DIR,
        mix=WorkerSFTMixConfig(total_examples=total_examples),
        model=WorkerSFTModelConfig(
            name="fake/base-model",
            device="cpu",
            dtype="float32",
            max_input_tokens=max_input_tokens,
        ),
        sft=WorkerSFTTrainConfig(
            num_epochs=num_epochs,
            learning_rate=1e-3,
            grad_accum_steps=grad_accum_steps,
        ),
        decode=WorkerSFTDecodeConfig(max_new_tokens=16),
        review=WorkerSFTReviewConfig(
            require_reviewed_for_training=require_reviewed
        ),
    )


def write_prepared_data(
    cfg: WorkerSFTConfig,
    train: List[WorkerExample],
    val: Optional[List[WorkerExample]] = None,
    test: Optional[List[WorkerExample]] = None,
    probes: Optional[List[WorkerExample]] = None,
) -> Path:
    """Write prepared_*.jsonl files (what the trainer loads)."""
    data_dir = cfg.prepared_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in (
        ("prepared_train", train),
        ("prepared_val", val or []),
        ("prepared_test", test or []),
        ("probes", probes or []),
    ):
        with open(data_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example.to_dict()) + "\n")
    return data_dir


def write_trace_support_files(
    cfg: WorkerSFTConfig,
    dedup_report: Optional[Dict[str, Any]] = None,
    label_quality: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the two report files the trace builder loads."""
    data_dir = cfg.prepared_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    default_dedup = {
        "methods": {"test": "fixture"},
        "removed": {},
        "counts": {},
        "eval_manifests": [],
    }
    default_quality = {
        "per_source": {},
        "teacher_file": {"status": "missing", "path": ""},
        "reviewed_file_problems": [],
    }
    (data_dir / "dedup_report.json").write_text(
        json.dumps(dedup_report or default_dedup), encoding="utf-8"
    )
    (data_dir / "label_quality_report.json").write_text(
        json.dumps(label_quality or default_quality), encoding="utf-8"
    )


def make_fake_model_for(
    examples: List[WorkerExample],
    hidden: int = 8,
) -> Any:
    """Fake model whose tokenizer knows every token of ``examples``."""
    return build_fake_model(corpus=corpus_for(examples), hidden=hidden)


def minimal_worker_sft_trace() -> Dict[str, Any]:
    """A smallest trace the renderer accepts (no invented numbers)."""
    return {
        "schema_version": 1,
        "A_run_identity": {
            "date": "2026-10-12T00:00:00+00:00",
            "run_id": "worker_sft-20261012T000000Z",
            "code_revision": "test-revision",
            "model": {
                "name": "fake/worker-model",
                "revision": None,
                "real_model": False,
                "fresh_adapter": True,
                "loaded_previous_adapter": False,
            },
            "hardware": {
                "platform": "test",
                "gpu_name": None,
                "python_version": "3.11",
            },
            "seeds": {
                "seed": 0,
                "sft_seed": 0,
                "example_order_seed": 0,
                "probe_seed": 12345,
            },
            "effective_config": {
                "sft": {
                    "num_epochs": 1,
                    "learning_rate": 1e-5,
                    "weight_decay": 0.0,
                    "microbatch_size": 1,
                    "grad_accum_steps": 1,
                    "max_grad_norm": 1.0,
                    "scheduler": "none",
                },
                "model": {
                    "peft": {"r": 16, "alpha": 32, "dropout": 0.05}
                },
            },
        },
        "B_raw_data": {"training_examples": [], "probe_examples": []},
        "C_data_conversion": {"examples": []},
        "D_dedup_and_selection": {
            "methods": {},
            "removed": {},
            "counts": {},
            "eval_manifests": [],
            "selected_training_ids": [],
        },
        "E_inputs_targets": {"examples": []},
        "F_tokenization": {
            "masking_rule": "prompt masked; assistant turn supervised",
            "examples": {},
        },
        "G_loss": {
            "teacher_forcing": "plain English explanation",
            "per_example": [],
            "update_losses": [],
        },
        "H_parameter_update": {
            "per_step": [],
            "optimizer": {},
            "trainable": {},
            "frozen_base_check": {
                "method": "sha256 before/after",
                "hash_before": "aaa",
                "hash_after": "aaa",
                "unchanged": True,
            },
        },
        "I_generation": {
            "decode": {"do_sample": False, "max_new_tokens": 128},
            "note": "same inputs before and after training",
            "inputs": [],
        },
        "J_limits": {
            "confirmed": ["pipeline ran"],
            "not_established": ["no eval in trace mode"],
            "deferred": ["smoke and normal runs"],
        },
    }
