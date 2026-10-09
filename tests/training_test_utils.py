"""Shared builders for the training test suite (not collected by pytest:
the filename does not match the ``test_*.py`` pattern).

Everything here runs offline: fixture HotpotQA rows, ``FakePolicy`` /
``FakeSynthesizer`` doubles, and ``TrainingConfig`` instances pointing at
temporary directories plus the real ``prompts_training/`` templates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from hotpot_mas.question_selection import (
    SelectedQuestion,
    build_manifest,
    select_questions,
)
from hotpot_mas.training.config import (
    DataConfig,
    SplitConfig,
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
        k = counts[question] - 1
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
