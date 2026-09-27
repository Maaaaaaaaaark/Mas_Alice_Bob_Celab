"""Shared fixtures: sample question, config, prompts, mock-run helper."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hotpot_mas.config import (  # noqa: E402
    ExperimentConfig,
    WorkerClarificationParams,
)
from hotpot_mas.model_engine import MockEngine  # noqa: E402
from hotpot_mas.orchestrator import Orchestrator  # noqa: E402
from hotpot_mas.prompts import PromptSet  # noqa: E402

# Distinctive markers embedded in each side's private evidence, used by the
# isolation tests to detect leakage. Neither appears in prompts or questions.
E_A_MARKER = "E_A_SECRET_MARKER"
E_B_MARKER = "E_B_SECRET_MARKER"

SAMPLE_QUESTION: Dict[str, Any] = {
    "question_id": "test_q0001",
    "question": "Which bridge connects Alpha City and Beta City?",
    "answer": "The Connector Bridge",
    "q_type": "bridge",
    "supporting_titles": ["Alpha City", "Beta City"],
    "evidence_alice": (
        "Title: Alpha City\n"
        "Alpha City lies on the west bank of the river. "
        f"{E_A_MARKER}: its landmark bridge opened in 1883."
    ),
    "evidence_bob": (
        "Title: Beta City\n"
        "Beta City lies on the east bank of the river. "
        f"{E_B_MARKER}: the bridge to Alpha City is called The Connector Bridge."
    ),
    "context_titles": ["Alpha City", "Beta City", "Unrelated City"],
}

# Scripts for a natural 3-step run: ask alice, ask bob, then final.
NATURAL_SCRIPTS: Dict[str, List[str]] = {
    "celab": [
        "Celab: <TO>ALICE</TO> Tell me about Alpha City.",
        "Celab: <TO>BOB</TO> Tell me about Beta City.",
        "Celab: <FINAL>The Connector Bridge</FINAL>",
    ],
    "alice": ["Alice: Alpha City is on the west bank of the river."],
    "bob": ["Bob: Beta City is on the east bank of the river."],
}


@pytest.fixture
def sample_question() -> Dict[str, Any]:
    return dict(SAMPLE_QUESTION)


@pytest.fixture
def prompts() -> PromptSet:
    return PromptSet(REPO_ROOT / "prompts")


@pytest.fixture
def base_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        output_dir=tmp_path / "outputs",
        manifest_path=tmp_path / "question_manifest.json",
        modes={},
    )


@pytest.fixture
def run_with_mock(
    prompts: PromptSet, base_config: ExperimentConfig
) -> Callable[..., Any]:
    """Run one trajectory against a MockEngine; returns (record, engine)."""

    def _run(
        scripts: Optional[Dict[str, List[Any]]] = None,
        cap_speakers: Optional[List[str]] = None,
        max_decision_steps: int = 20,
        question: Optional[Dict[str, Any]] = None,
        run_seed: int = 0,
        clarification_enabled: bool = False,
        clarification_max_per_worker: int = 1,
    ) -> Any:
        engine = MockEngine(scripts=scripts or dict(NATURAL_SCRIPTS), cap_speakers=cap_speakers)
        cfg = replace(
            base_config,
            max_decision_steps=max_decision_steps,
            worker_clarification=WorkerClarificationParams(
                enabled=clarification_enabled,
                max_per_worker=clarification_max_per_worker,
            ),
        )
        orchestrator = Orchestrator(
            cfg, prompts, engine, {"environment": "mock-test"}
        )
        record = orchestrator.run_one(
            question or dict(SAMPLE_QUESTION), 0, run_seed, "test_q0001-run-00"
        )
        return record, engine

    return _run
