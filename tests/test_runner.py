"""Runner identity and safe-resume tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hotpot_mas.config import GenerationParams
from hotpot_mas.model_engine import MockEngine
from hotpot_mas.runner import run_experiment

from conftest import NATURAL_SCRIPTS, SAMPLE_QUESTION


def test_resume_skips_only_matching_fingerprint(base_config, prompts, tmp_path: Path):
    cfg = replace(base_config, runs_per_question=1, run_seeds=[7])
    runs_path = tmp_path / "runs.jsonl"
    first_engine = MockEngine(scripts=dict(NATURAL_SCRIPTS))
    completed = run_experiment(
        cfg, [dict(SAMPLE_QUESTION)], first_engine, {}, prompts,
        runs_path, skip_report=True,
    )
    assert completed[0]["run_id"].endswith("-seed-7")
    assert len(completed[0]["run_fingerprint"]) == 64

    empty_engine = MockEngine()
    assert run_experiment(
        cfg, [dict(SAMPLE_QUESTION)], empty_engine, {}, prompts,
        runs_path, skip_report=False,
    ) == []
    assert empty_engine.calls == []
    assert (tmp_path / "report.md").is_file()

    changed = replace(
        cfg,
        generation=GenerationParams(
            do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=2048
        ),
    )
    with pytest.raises(RuntimeError, match="different or missing run_fingerprint"):
        run_experiment(
            changed, [dict(SAMPLE_QUESTION)], MockEngine(), {}, prompts,
            runs_path, skip_report=True,
        )
