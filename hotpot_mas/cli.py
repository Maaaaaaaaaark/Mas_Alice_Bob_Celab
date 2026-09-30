"""Command-line entry point (spec sec. 19).

Subcommands:

- ``prepare-questions``: download HotpotQA validation and write the fixed
  question manifest (needs network + ``datasets``);
- ``run``: execute trajectories (needs a GPU + the gated model);
- ``report``: (re)generate summaries/report.md from an existing runs.jsonl.

Usage: ``python -m hotpot_mas.cli <subcommand> [options]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import ExperimentConfig
from .diagnostics import write_diagnostic_report
from .question_selection import build_manifest
from .report import write_report


def _load_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(args.config)
    cfg = cfg.apply_mode(
        mode=getattr(args, "mode", None),
        num_questions=getattr(args, "num_questions", None),
        runs_per_question=getattr(args, "runs_per_question", None),
        run_seeds=getattr(args, "run_seeds", None),
    )
    return cfg


def _cmd_prepare_questions(args: argparse.Namespace) -> None:
    from .hotpotqa import load_hotpotqa_validation

    cfg = _load_config(args)
    print(f"loading HotpotQA {cfg.dataset}/{cfg.dataset_config} "
          f"{cfg.dataset_split} ...")
    rows = load_hotpotqa_validation(
        dataset_config=cfg.dataset_config,
        dataset_name=cfg.dataset,
        split=cfg.dataset_split,
    )
    print(f"loaded {len(rows)} rows")
    manifest = build_manifest(
        rows,
        num_questions=cfg.num_questions,
        selection_seed=cfg.sample_selection_seed,
        manifest_path=cfg.manifest_path,
        evidence_partition=cfg.evidence_partition,
        partition_seed=cfg.partition_seed,
    )
    print(
        f"manifest written to {cfg.manifest_path}: "
        f"{manifest['num_questions_selected']} questions "
        f"(sha256 {manifest['questions_sha256'][:16]}...)"
    )
    print("excluded statistics:")
    for reason, count in sorted(manifest["excluded_statistics"].items()):
        print(f"  {reason}: {count}")


def _cmd_run(args: argparse.Namespace) -> None:
    cfg = _load_config(args)
    from .runner import run_from_config

    run_from_config(cfg)


def _cmd_report(args: argparse.Namespace) -> None:
    runs_path = Path(args.runs)
    if not runs_path.is_absolute():
        from .config import REPO_ROOT

        runs_path = REPO_ROOT / runs_path
    paths = write_report(runs_path, runs_path.parent)
    print("report files written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def _cmd_diagnose(args: argparse.Namespace) -> None:
    from .config import REPO_ROOT

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path

    paths = write_diagnostic_report(
        resolve(args.baseline),
        resolve(args.clarification),
        resolve(args.centralized),
        resolve(args.output_dir),
        shared_question_path=(
            resolve(args.shared_question) if args.shared_question else None
        ),
        one_shot_path=(resolve(args.one_shot) if args.one_shot else None),
    )
    print("diagnostic files written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hotpot_mas.cli",
        description="HotpotQA 3-agent MAS base experiment",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--config", default="configs/base.yaml", help="YAML config path"
        )

    prep = subparsers.add_parser(
        "prepare-questions",
        help="download HotpotQA validation and write the question manifest",
    )
    add_config(prep)
    prep.add_argument(
        "--mode",
        help="optional preparation mode from the config",
    )
    prep.add_argument(
        "--questions", dest="num_questions", type=int,
        help="override the number of questions written to the manifest",
    )
    prep.set_defaults(func=_cmd_prepare_questions)

    run = subparsers.add_parser(
        "run", help="run trajectories (append to runs.jsonl, resumable)"
    )
    add_config(run)
    run.add_argument(
        "--mode",
        choices=["smoke", "small", "pilot", "benchmark", "full"],
        help="run mode from the config (overrides questions/runs/seeds)",
    )
    run.add_argument(
        "--questions", dest="num_questions", type=int,
        help="override num_questions",
    )
    run.add_argument(
        "--runs", dest="runs_per_question", type=int,
        help="override runs_per_question (uses the first N configured seeds)",
    )
    run.add_argument(
        "--seeds", dest="run_seeds",
        help="comma-separated run seed list (0,1,2)"
    )
    run.set_defaults(func=_cmd_run)

    rep = subparsers.add_parser(
        "report", help="(re)generate summaries and report.md from runs.jsonl"
    )
    rep.add_argument("--runs", required=True, help="path to runs.jsonl")
    rep.set_defaults(func=_cmd_report)

    diagnose = subparsers.add_parser(
        "diagnose",
        help="compare matched MAS controls and centralized-reader runs",
    )
    diagnose.add_argument("--baseline", required=True, help="baseline runs.jsonl")
    diagnose.add_argument(
        "--clarification", required=True, help="clarification runs.jsonl"
    )
    diagnose.add_argument(
        "--centralized", required=True, help="centralized-reader runs.jsonl"
    )
    diagnose.add_argument(
        "--shared-question", help="shared-question MAS runs.jsonl"
    )
    diagnose.add_argument(
        "--one-shot", help="one-shot gather MAS runs.jsonl"
    )
    diagnose.add_argument(
        "--output-dir",
        default="outputs/hotpotqa_diagnostics/reading_vs_communication",
        help="directory for the English diagnostic report and JSON summary",
    )
    diagnose.set_defaults(func=_cmd_diagnose)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
