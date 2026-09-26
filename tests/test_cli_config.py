"""CLI override and run-count invariant tests."""

from __future__ import annotations

import pytest

from hotpot_mas.cli import _build_parser, _load_config


def _parse(*extra: str):
    return _build_parser().parse_args(
        ["run", "--config", "configs/base.yaml", *extra]
    )


def test_cli_override_destinations_are_applied():
    cfg = _load_config(
        _parse("--mode", "full", "--questions", "3", "--runs", "2")
    )
    assert cfg.num_questions == 3
    assert cfg.runs_per_question == 2
    assert cfg.run_seeds == [0, 1]


def test_seed_override_sets_run_count():
    cfg = _load_config(_parse("--seeds", "7,8,9"))
    assert cfg.runs_per_question == 3
    assert cfg.run_seeds == [7, 8, 9]


def test_mismatched_runs_and_seeds_are_rejected():
    with pytest.raises(ValueError, match="must equal len"):
        _load_config(_parse("--runs", "2", "--seeds", "7,8,9"))

