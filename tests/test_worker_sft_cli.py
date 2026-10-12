"""CLI wiring tests for the Stage 1a subcommands. CPU-only; the only
end-to-end path exercised here is the render command, which needs no
model, no tokenizer, and no prepared data."""

from __future__ import annotations

import json

import pytest

from hotpot_mas.training.cli import build_parser, main

from worker_sft_test_utils import minimal_worker_sft_trace


def test_parser_exposes_the_four_worker_sft_subcommands():
    parser = build_parser()

    args = parser.parse_args(
        ["prepare-worker-sft-data", "--config", "x.yaml", "--mode", "trace"]
    )
    assert args.command == "prepare-worker-sft-data"
    assert args.mode == "trace"

    args = parser.parse_args(
        ["inspect-worker-sft-lengths", "--config", "x.yaml", "--mode", "smoke"]
    )
    assert args.command == "inspect-worker-sft-lengths"

    args = parser.parse_args(
        ["train-worker-sft", "--config", "x.yaml", "--mode", "normal",
         "--resume"]
    )
    assert args.command == "train-worker-sft"
    assert args.resume is True

    args = parser.parse_args(
        ["render-worker-sft-trace", "--trace", "trace.json"]
    )
    assert args.command == "render-worker-sft-trace"
    assert args.trace == "trace.json"


def test_worker_sft_mode_choices_are_trace_smoke_normal_only():
    parser = build_parser()
    for subcommand in (
        "prepare-worker-sft-data",
        "inspect-worker-sft-lengths",
        "train-worker-sft",
    ):
        # Stage 1 / Stage 2 modes must NOT leak into the Stage 1a CLI.
        with pytest.raises(SystemExit):
            parser.parse_args(
                [subcommand, "--config", "x.yaml", "--mode", "overfit"]
            )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["train-worker-sft", "--config", "x.yaml", "--mode", "full"]
        )


def test_render_worker_sft_trace_command_end_to_end(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(minimal_worker_sft_trace()), encoding="utf-8"
    )
    out_path = tmp_path / "rendered" / "worked.md"
    assert (
        main(
            [
                "render-worker-sft-trace",
                "--trace",
                str(trace_path),
                "--output",
                str(out_path),
            ]
        )
        == 0
    )
    md = out_path.read_text(encoding="utf-8")
    assert "## A. Run identity" in md
    assert "## J. Limits and next steps" in md


def test_render_worker_sft_trace_defaults_output_next_to_trace(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(minimal_worker_sft_trace()), encoding="utf-8"
    )
    assert main(["render-worker-sft-trace", "--trace", str(trace_path)]) == 0
    default = trace_path.parent / "worked_example_worker_sft.md"
    assert default.is_file()


def test_render_worker_sft_trace_fails_loudly_on_missing_section(tmp_path):
    trace = minimal_worker_sft_trace()
    del trace["G_loss"]
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="missing section"):
        main(["render-worker-sft-trace", "--trace", str(trace_path)])
