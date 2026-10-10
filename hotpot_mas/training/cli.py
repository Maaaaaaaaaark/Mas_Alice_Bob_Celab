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
    outcome = InferenceDiagnostic(cfg).run()
    print("inference diagnostic finished:")
    for key, value in outcome.items():
        print(f"  {key}: {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hotpot_mas.training.cli",
        description="Cross-paired GRPO training (Stage 1 workers).",
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
        "train-workers", help="Stage 1: train the shared worker LoRA"
    )
    train.add_argument("--config", required=True)
    train.add_argument(
        "--mode", required=True, choices=["trace", "smoke", "pilot", "full"]
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
    }
    try:
        return handlers[args.command](args)
    except Exception as exc:  # noqa: BLE001 - report and fail loudly
        logger.error("command %s failed: %s", args.command, exc)
        raise


if __name__ == "__main__":
    sys.exit(main())
