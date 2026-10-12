"""Training CLI: ``python -m hotpot_mas.training.cli``.

Subcommands:

- ``prepare-data``: build the three fixed manifests + ``splits.json`` for
  one configuration (or one mode override of it).
- ``train-workers``: Stage 1 cross-paired GRPO worker training in the
  requested run mode (trace / smoke / pilot / full).
- ``train-synthesizer``: Stage 2 SFT of the synthesizer C against the
  frozen best Stage 1 workers (trace / smoke).
- ``render-trace``: regenerate ``worked_example.md`` from an existing
  ``trace.json`` (no model, no GPU).
- ``diagnose-inference``: run the four-condition validation diagnostic
  without training or loading a trained checkpoint.
- ``prepare-worker-sft-data``: Stage 1a data prep (conversion, label
  checks, dedup, mixture selection) into ``outputs/worker_sft_data/``.
- ``inspect-worker-sft-lengths``: tokenizer-only length report for the
  prepared Stage 1a examples (no model weights, no training).
- ``train-worker-sft``: Stage 1a SFT of the shared worker LoRA
  (trace / smoke / normal).
- ``render-worker-sft-trace``: regenerate
  ``worked_example_worker_sft.md`` from a Stage 1a ``trace.json``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("hotpot_mas.training.cli")


def cmd_prepare_data(args: argparse.Namespace) -> int:
    from .config import TrainingConfig
    from .data import describe_split, prepare_manifests

    cfg = TrainingConfig.from_yaml(args.config, mode=args.mode)
    print(describe_split(cfg))
    paths = prepare_manifests(cfg, force=args.force)
    for split, path in paths.items():
        print(f"  manifest {split}: {path}")
    print(f"  splits: {cfg.manifest_dir / 'splits.json'}")
    return 0


def cmd_train_workers(args: argparse.Namespace) -> int:
    from .config import TrainingConfig
    from .data import describe_split, load_splits
    from .worker_trainer import WorkerTrainer

    cfg = TrainingConfig.from_yaml(args.config, mode=args.mode)
    print(describe_split(cfg))
    # Fail fast with a clear message when the manifests are missing.
    load_splits(cfg.manifest_dir)
    trainer = WorkerTrainer(cfg, mode=args.mode, resume=args.resume)
    outcome = trainer.train()
    print("training finished:")
    for key, value in outcome.items():
        print(f"  {key}: {value}")
    return 0


def cmd_train_synthesizer(args: argparse.Namespace) -> int:
    from .config import SynthesizerTrainingConfig
    from .data import describe_split, load_splits
    from .synthesizer_trainer import SynthesizerTrainer

    cfg = SynthesizerTrainingConfig.from_yaml(args.config, mode=args.mode)
    print(describe_split(cfg))
    # Fail fast with a clear message when the manifests are missing.
    load_splits(cfg.manifest_dir)
    trainer = SynthesizerTrainer(cfg, mode=args.mode)
    outcome = trainer.train()
    print("synthesizer SFT finished:")
    for key, value in outcome.items():
        if key != "comparison":
            print(f"  {key}: {value}")
    return 0


def cmd_render_trace(args: argparse.Namespace) -> int:
    from .trace import render_worked_example

    trace_path = Path(args.trace)
    output_path = (
        Path(args.output)
        if args.output
        else trace_path.parent / "worked_example.md"
    )
    render_worked_example(trace_path, output_path)
    print(f"worked example written: {output_path}")
    return 0


def cmd_diagnose_inference(args: argparse.Namespace) -> int:
    from .config import TrainingConfig
    from .data import describe_split, load_splits
    from .inference_diagnostic import InferenceDiagnostic

    cfg = TrainingConfig.from_yaml(args.config, mode=args.mode)
    print(describe_split(cfg))
    load_splits(cfg.manifest_dir)
    report_cache = Path(args.report_cache) if args.report_cache else None
    baseline_summary = (
        Path(args.baseline_summary) if args.baseline_summary else None
    )
    outcome = InferenceDiagnostic(
        cfg,
        report_cache_path=report_cache,
        baseline_summary_path=baseline_summary,
    ).run()
    print("inference diagnostic finished:")
    for key, value in outcome.items():
        print(f"  {key}: {value}")
    return 0


def cmd_check_logprobs(args: argparse.Namespace) -> int:
    from .config import TrainingConfig
    from .data import describe_split, load_splits
    from .learner_checks import check_logprob_consistency

    cfg = TrainingConfig.from_yaml(args.config, mode=args.mode)
    print(describe_split(cfg))
    load_splits(cfg.manifest_dir)
    result = check_logprob_consistency(
        cfg,
        mean_tolerance=args.mean_tolerance,
        max_tolerance=args.max_tolerance,
    )
    print("log-prob consistency passed:")
    for key, value in result.items():
        if key != "sides":
            print(f"  {key}: {value}")
    return 0


def cmd_compare_rewards(args: argparse.Namespace) -> int:
    from .config import TrainingConfig
    from .data import describe_split, load_splits
    from .reward_comparison import RewardComparison

    cfg = TrainingConfig.from_yaml(args.config, mode=args.mode)
    print(describe_split(cfg))
    load_splits(cfg.manifest_dir)
    outcome = RewardComparison(cfg).run()
    print("reward comparison finished:")
    for key, value in outcome.items():
        print(f"  {key}: {value}")
    return 0


def cmd_prepare_worker_sft_data(args: argparse.Namespace) -> int:
    from .worker_sft.config import WorkerSFTConfig
    from .worker_sft.runner import prepare_worker_sft_data

    cfg = WorkerSFTConfig.from_yaml(args.config, mode=args.mode)
    paths = prepare_worker_sft_data(cfg, args.mode, force=args.force)
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def cmd_inspect_worker_sft_lengths(args: argparse.Namespace) -> int:
    from .worker_sft.config import WorkerSFTConfig
    from .worker_sft.inspect import inspect_worker_sft_lengths

    cfg = WorkerSFTConfig.from_yaml(args.config, mode=args.mode)
    report_path = inspect_worker_sft_lengths(cfg)
    print(f"  lengths report: {report_path}")
    return 0


def cmd_train_worker_sft(args: argparse.Namespace) -> int:
    from .worker_sft.config import WorkerSFTConfig
    from .worker_sft.trainer import run_worker_sft_training

    cfg = WorkerSFTConfig.from_yaml(args.config, mode=args.mode)
    outcome = run_worker_sft_training(
        cfg, mode=args.mode, resume=args.resume
    )
    print("worker SFT finished:")
    for key, value in outcome.items():
        print(f"  {key}: {value}")
    return 0


def cmd_render_worker_sft_trace(args: argparse.Namespace) -> int:
    from .worker_sft.trace import render_worker_sft_worked_example

    trace_path = Path(args.trace)
    output_path = (
        Path(args.output)
        if args.output
        else trace_path.parent / "worked_example_worker_sft.md"
    )
    render_worker_sft_worked_example(trace_path, output_path)
    print(f"worked example written: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hotpot_mas.training.cli",
        description=(
            "Hotpot multi-agent training and diagnostics (legacy GRPO, "
            "synthesizer SFT, and current independent worker SFT)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser(
        "prepare-data",
        help="build train/val/test manifests + splits.json",
    )
    prepare.add_argument("--config", required=True)
    prepare.add_argument(
        "--mode",
        default=None,
        help="apply this declared mode override to the config",
    )
    prepare.add_argument(
        "--force", action="store_true",
        help="rebuild manifests even if splits.json exists",
    )

    train = sub.add_parser(
        "train-workers",
        help="legacy cross-paired GRPO worker training (uses frozen C)",
    )
    train.add_argument("--config", required=True)
    train.add_argument(
        "--mode",
        required=True,
        choices=[
            "trace",
            "smoke",
            "pilot",
            "full",
            "overfit",
            "overfit_f1",
            "overfit_em",
            "overfit_gold_mean_log_likelihood",
        ],
    )
    train.add_argument(
        "--resume", action="store_true",
        help="resume from the latest checkpoint in the stage directory",
    )

    synth = sub.add_parser(
        "train-synthesizer",
        help="Stage 2: SFT of C against frozen best workers",
    )
    synth.add_argument("--config", required=True)
    synth.add_argument(
        "--mode", required=True, choices=["trace", "smoke"]
    )

    render = sub.add_parser(
        "render-trace",
        help="regenerate worked_example.md from a trace.json",
    )
    render.add_argument("--trace", required=True)
    render.add_argument("--output", default=None)

    diagnostic = sub.add_parser(
        "diagnose-inference",
        help="inference-only four-condition validation diagnostic",
    )
    diagnostic.add_argument("--config", required=True)
    diagnostic.add_argument(
        "--mode",
        default=None,
        help="apply this declared mode override to the config",
    )
    diagnostic.add_argument(
        "--report-cache",
        default=None,
        help="reuse an existing untrained-worker report cache",
    )
    diagnostic.add_argument(
        "--baseline-summary",
        default=None,
        help="compare the new F1 ordering with a previous summary.json",
    )

    consistency = sub.add_parser(
        "check-logprobs",
        help="fail-fast sampling versus teacher-forcing log-prob check",
    )
    consistency.add_argument("--config", required=True)
    consistency.add_argument("--mode", default=None)
    consistency.add_argument("--mean-tolerance", type=float, default=2.0e-3)
    consistency.add_argument("--max-tolerance", type=float, default=2.0e-2)

    rewards = sub.add_parser(
        "compare-rewards",
        help="compare EM, F1, and gold-answer likelihood rewards",
    )
    rewards.add_argument("--config", required=True)
    rewards.add_argument("--mode", default=None)

    prepare_worker = sub.add_parser(
        "prepare-worker-sft-data",
        help="Stage 1a: convert, check, dedup, and select worker data",
    )
    prepare_worker.add_argument("--config", required=True)
    prepare_worker.add_argument(
        "--mode",
        required=True,
        choices=["trace", "smoke", "normal"],
    )
    prepare_worker.add_argument(
        "--force", action="store_true",
        help="rebuild prepared data even if it already exists",
    )

    inspect = sub.add_parser(
        "inspect-worker-sft-lengths",
        help="Stage 1a: tokenizer-only length report (no model weights)",
    )
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--mode", required=True,
                         choices=["trace", "smoke", "normal"])

    train_worker = sub.add_parser(
        "train-worker-sft",
        help="Stage 1a: SFT of the shared worker LoRA",
    )
    train_worker.add_argument("--config", required=True)
    train_worker.add_argument(
        "--mode", required=True, choices=["trace", "smoke", "normal"]
    )
    train_worker.add_argument(
        "--resume", action="store_true",
        help="resume from this run's own checkpoint in the stage directory",
    )

    render_worker = sub.add_parser(
        "render-worker-sft-trace",
        help="regenerate worked_example_worker_sft.md from a trace.json",
    )
    render_worker.add_argument("--trace", required=True)
    render_worker.add_argument("--output", default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "prepare-data": cmd_prepare_data,
        "train-workers": cmd_train_workers,
        "train-synthesizer": cmd_train_synthesizer,
        "render-trace": cmd_render_trace,
        "diagnose-inference": cmd_diagnose_inference,
        "check-logprobs": cmd_check_logprobs,
        "compare-rewards": cmd_compare_rewards,
        "prepare-worker-sft-data": cmd_prepare_worker_sft_data,
        "inspect-worker-sft-lengths": cmd_inspect_worker_sft_lengths,
        "train-worker-sft": cmd_train_worker_sft,
        "render-worker-sft-trace": cmd_render_worker_sft_trace,
    }
    try:
        return handlers[args.command](args)
    except Exception as exc:  # noqa: BLE001 - report and fail loudly
        logger.error("command %s failed: %s", args.command, exc)
        raise


if __name__ == "__main__":
    sys.exit(main())
