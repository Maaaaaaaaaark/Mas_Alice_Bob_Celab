"""CLI parsing tests: Stage 2 ``train-synthesizer`` plus Stage 1 regression.

Parser-level only — no model loading happens here.
"""

from __future__ import annotations

import pytest

from hotpot_mas.training.cli import build_parser


def parse(argv):
    return build_parser().parse_args(argv)


def test_train_synthesizer_trace() -> None:
    args = parse(
        ["train-synthesizer", "--config", "configs/synthesizer_sft.yaml",
         "--mode", "trace"]
    )
    assert args.command == "train-synthesizer"
    assert args.config == "configs/synthesizer_sft.yaml"
    assert args.mode == "trace"


def test_train_synthesizer_smoke() -> None:
    args = parse(
        ["train-synthesizer", "--config", "configs/synthesizer_sft.yaml",
         "--mode", "smoke"]
    )
    assert args.command == "train-synthesizer"
    assert args.mode == "smoke"


def test_train_synthesizer_requires_config() -> None:
    with pytest.raises(SystemExit):
        parse(["train-synthesizer", "--mode", "trace"])


def test_train_synthesizer_requires_mode() -> None:
    with pytest.raises(SystemExit):
        parse(["train-synthesizer", "--config", "x.yaml"])


def test_train_synthesizer_rejects_unknown_mode() -> None:
    with pytest.raises(SystemExit):
        parse(
            ["train-synthesizer", "--config", "x.yaml", "--mode", "full"]
        )


def test_train_workers_choices_unchanged() -> None:
    for mode in ("trace", "smoke", "pilot", "full"):
        args = parse(
            ["train-workers", "--config", "x.yaml", "--mode", mode]
        )
        assert args.command == "train-workers"
        assert args.mode == mode


def test_train_workers_still_rejects_stage2_modes() -> None:
    # The Stage 1 subcommand must not accept Stage 2 modes and vice versa.
    with pytest.raises(SystemExit):
        parse(["train-workers", "--config", "x.yaml", "--mode", "smoke2"])
    args = parse(
        ["train-workers", "--config", "x.yaml", "--mode", "smoke",
         "--resume"]
    )
    assert args.resume is True


def test_prepare_data_and_render_trace_unchanged() -> None:
    args = parse(
        ["prepare-data", "--config", "x.yaml", "--mode", "smoke",
         "--force"]
    )
    assert args.command == "prepare-data"
    assert args.force is True
    args = parse(["render-trace", "--trace", "t.json"])
    assert args.command == "render-trace"
    assert args.trace == "t.json"
