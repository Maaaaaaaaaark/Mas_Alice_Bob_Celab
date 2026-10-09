"""Shared builders for the training test suite (not collected by pytest:
the filename does not match the ``test_*.py`` pattern).

Everything here runs offline: fixture HotpotQA rows, ``FakePolicy`` /
``FakeSynthesizer`` doubles, and ``TrainingConfig`` instances pointing at
temporary directories plus the real ``prompts_training/`` templates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from hotpot_mas.question_selection import (
    SelectedQuestion,
    build_manifest,
    select_questions,
)
from hotpot_mas.training.config import (
    DataConfig,
    DecodeConfig,
    ModelConfig,
    SFTConfig,
    SplitConfig,
    SynthesizerTrainingConfig,
    TrainingConfig,
    WorkerTrainingConfig,
)
from hotpot_mas.training.fake_policy import (
    FakePolicy,
    FakeSynthesizer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_TRAINING_DIR = REPO_ROOT / "prompts_training"

GOLD_ANSWER = "gold answer text"
# Deliberately zero token overlap with GOLD_ANSWER so the real F1 code
# scores it exactly 0.0 (a near-miss like "wrong answer" shares the token
# "answer" and would score 0.4, corrupting the scripted reward matrices).
WRONG_ANSWER = "banana"


def make_distractor_row(
    qid: str, answer: str = GOLD_ANSWER
) -> Dict[str, Any]:
    """One valid balanced_distractor candidate: 10 distinct docs, 2 gold."""
    supporting = [
        {"title": "Sup One", "sentences": ["Gold sentence one."]},
        {"title": "Sup Two", "sentences": ["Gold sentence two."]},
    ]
    distractors = [
        {
            "title": f"Dis {index}",
            "sentences": [f"Distractor sentence {index}."],
        }
        for index in range(8)
    ]
    context = supporting + distractors
    return {
        "id": qid,
        "question": f"Question text for {qid}?",
        "answer": answer,
        "type": "bridge",
        "level": "hard",
        "supporting_facts": [["Sup One", 0], ["Sup Two", 0]],
        "context": context,
    }


def make_rows(count: int, prefix: str = "q") -> List[Dict[str, Any]]:
    return [make_distractor_row(f"{prefix}{index}") for index in range(count)]


def select_balanced_questions(
    rows: List[Dict[str, Any]],
    num_questions: int,
    selection_seed: int,
    partition_seed: int = 0,
) -> List[SelectedQuestion]:
    return select_questions(
        rows,
        num_questions,
        selection_seed,
        evidence_partition="balanced_distractor",
        partition_seed=partition_seed,
    )


def build_balanced_manifest(
    rows: List[Dict[str, Any]],
    num_questions: int,
    selection_seed: int,
    manifest_path: Path,
    partition_seed: int = 0,
    exclude_ids: Optional[set] = None,
) -> Dict[str, Any]:
    return build_manifest(
        rows,
        num_questions,
        selection_seed,
        manifest_path,
        evidence_partition="balanced_distractor",
        partition_seed=partition_seed,
        exclude_ids=exclude_ids,
    )


def make_training_config(
    tmp_path: Path,
    *,
    train_num: int = 4,
    val_num: int = 2,
    test_num: int = 2,
    seed: int = 0,
    **worker_overrides: Any,
) -> TrainingConfig:
    """A config whose splits all come from the (mocked) validation source."""
    workers = dict(worker_overrides)
    return TrainingConfig(
        experiment_id="test_experiment",
        experiment_version="v1",
        seed=seed,
        output_dir=tmp_path / "outputs",
        manifest_dir=tmp_path / "manifests",
        prompt_dir=PROMPTS_TRAINING_DIR,
        data=DataConfig(
            train=SplitConfig("validation", train_num, 0),
            val=SplitConfig("validation", val_num, 1),
            test=SplitConfig("validation", test_num, 2),
            partition_seed=0,
            val_test_split_seed=0,
        ),
        workers=WorkerTrainingConfig(
            **{
                "G": 2,
                "questions_per_update": 1,
                "steps": 1,
                "eval_interval": 5,
                "final_test_eval": False,
                "max_sampling_attempts": 12,
                **workers,
            }
        ),
    )


def scripted_synthesizer(
    scripts: Dict[str, List[List[str]]], g: int
) -> Tuple[FakeSynthesizer, List[Tuple[str, str, str]]]:
    """Synthesizer whose answer for pair (i, j) is ``scripts[q][i][j]``.

    Call order within a question is i-major (matching rollout), tracked by
    a per-question counter. Returns the synthesizer and the raw call list.
    """
    counts: Dict[str, int] = {}
    calls: List[Tuple[str, str, str]] = []

    def answer_fn(question: str, a_text: str, b_text: str) -> str:
        calls.append((question, a_text, b_text))
        counts[question] = counts.get(question, 0) + 1
        # A question may be rolled out again on a later update.  Reuse the
        # same deterministic G x G script for each rollout.
        k = (counts[question] - 1) % (g * g)
        i, j = divmod(k, g)
        script = scripts.get(question)
        if script is None:
            return WRONG_ANSWER
        return script[i][j]

    return FakeSynthesizer(answer_fn), calls


def matrix_of(value: str) -> List[List[str]]:
    """A 2x2 scripted answer matrix filled with one prediction string."""
    return [[value, value], [value, value]]


# Scripted 2x2 answer matrices with known cross-paired structure.
GOLD = GOLD_ANSWER
WRONG = WRONG_ANSWER
MATRIX_A_ONLY = [[GOLD, GOLD], [WRONG, WRONG]]   # q_a std > 0, q_b flat
MATRIX_B_ONLY = [[GOLD, WRONG], [GOLD, WRONG]]   # q_b std > 0, q_a flat
MATRIX_BOTH = [[GOLD, WRONG], [WRONG, WRONG]]  # q_a and q_b both vary
MATRIX_NONE = matrix_of(WRONG)                 # flat: no signal


# ----------------------------------------------------------------------
# Stage 2 (synthesizer SFT) builders
# ----------------------------------------------------------------------

def make_fake_worker_run(
    tmp_path: Path,
    val_f1_by_step: Dict[int, float],
    steps: Optional[int] = None,
    policy: Optional[FakePolicy] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """A fake Stage 1 run directory for best-checkpoint resolution tests.

    Writes ``step_NNNNN/`` directories (each with a ``FakePolicy`` adapter
    and a ``state.json`` carrying the running best), plus a ``metrics.jsonl``
    with one line per update. Steps missing from ``val_f1_by_step`` record
    ``mean_val_f1: null`` (as a trace-like or skipped-eval step would).
    Returns the run dir and the expected ``{"best_step", "best_val_f1"}``.
    """
    run_dir = tmp_path / "cross_paired_grpo_workers" / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    policy = policy or FakePolicy()
    if steps is None:
        steps = max(val_f1_by_step)
    best_val_f1 = -1.0
    best_step: Optional[int] = None
    lines: List[str] = []
    for step in range(1, steps + 1):
        step_dir = run_dir / f"step_{step:05d}"
        policy.save_adapter(step_dir / "adapter")
        f1 = val_f1_by_step.get(step)
        if f1 is not None and f1 > best_val_f1:
            best_val_f1 = float(f1)
            best_step = step
        (step_dir / "state.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "best_val_f1": best_val_f1,
                    "best_step": best_step,
                    "q_pointer": 0,
                    "mode": "smoke",
                }
            ),
            encoding="utf-8",
        )
        lines.append(
            json.dumps(
                {
                    "run_id": f"update-{step:05d}",
                    "update": step,
                    "mode": "smoke",
                    "mean_val_f1": f1,
                    "mean_val_em": 0.0,
                    "best_val_f1": best_val_f1,
                    "checkpoint_path": str(step_dir),
                }
            )
        )
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return run_dir, {"best_step": best_step, "best_val_f1": best_val_f1}


def make_synthesizer_training_config(
    tmp_path: Path,
    worker_checkpoint: Any,
    *,
    train_num: int = 4,
    val_num: int = 2,
    test_num: int = 2,
    seed: int = 0,
    num_epochs: int = 2,
    batch_size: int = 4,
    worker_eval_decode: Optional[DecodeConfig] = None,
    **sft_overrides: Any,
) -> SynthesizerTrainingConfig:
    """A Stage 2 config pointing at a fake worker run and temp dirs."""
    sft_kwargs: Dict[str, Any] = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        **sft_overrides,
    }
    if worker_eval_decode is not None:
        sft_kwargs["worker_eval_decode"] = worker_eval_decode
    return SynthesizerTrainingConfig(
        experiment_id="test_synthesizer_sft",
        experiment_version="v1",
        seed=seed,
        output_dir=tmp_path / "outputs",
        manifest_dir=tmp_path / "manifests",
        prompt_dir=PROMPTS_TRAINING_DIR,
        data=DataConfig(
            train=SplitConfig("validation", train_num, 0),
            val=SplitConfig("validation", val_num, 1),
            test=SplitConfig("validation", test_num, 2),
            partition_seed=0,
            val_test_split_seed=0,
        ),
        model=ModelConfig(
            name="fake/base-model",
            device="cpu",
            dtype="float32",
            max_input_length=30000,
        ),
        sft=SFTConfig(**sft_kwargs),
        worker_checkpoint=str(worker_checkpoint),
    )


def build_stage2_manifests(
    tmp_path: Path,
    *,
    train_num: int = 4,
    val_num: int = 2,
    test_num: int = 2,
) -> Dict[str, Path]:
    """Disjoint train/val/test manifests for the synthesizer trainer tests.

    The three splits use distinct question-id prefixes, so the manifests
    are pairwise disjoint without any exclusion bookkeeping.
    """
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        ("train", "t", train_num, 0),
        ("val", "v", val_num, 1),
        ("test", "e", test_num, 2),
    )
    paths: Dict[str, Path] = {}
    for name, prefix, count, selection_seed in specs:
        path = manifest_dir / f"{name}.json"
        build_balanced_manifest(
            make_rows(count, prefix=prefix),
            count,
            selection_seed,
            path,
        )
        paths[name] = path
    return paths
