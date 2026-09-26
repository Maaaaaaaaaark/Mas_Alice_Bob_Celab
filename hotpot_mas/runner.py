"""Trajectory loop: load config + manifest, run questions x seeds, log.

Runs are written to ``runs.jsonl`` one line at a time (append mode) so an
interrupted run can be resumed: existing ``run_id`` values are skipped.
Each run's progress is printed to stdout via ``tqdm.write``. The report is
written automatically at the end of a non-empty run set.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import tqdm

from .config import ExperimentConfig
from .logging_io import JsonlWriter, collect_environment_info
from .model_engine import HFEngine
from .orchestrator import Orchestrator
from .prompts import PromptSet
from .question_selection import load_manifest
from .report import write_report


def build_engine(cfg: ExperimentConfig) -> HFEngine:
    """Construct the shared frozen model engine (one instance per process)."""
    return HFEngine(
        model_name=cfg.model_name,
        generation_params=cfg.generation,
        dtype=cfg.dtype,
        device=cfg.device,
        attn_implementation=cfg.attn_implementation,
        model_revision=cfg.model_revision,
        tokenizer_revision=cfg.tokenizer_revision,
        max_input_length=cfg.max_input_length,
    )


def run_experiment(
    cfg: ExperimentConfig,
    questions: List[Any],
    engine: HFEngine,
    environment_info: Dict[str, Any],
    prompts: PromptSet,
    runs_path: Path,
    skip_report: bool = False,
) -> List[Dict[str, Any]]:
    """Run all (question, seed) pairs, appending each record to runs_path."""
    orchestrator = Orchestrator(cfg, prompts, engine, environment_info)
    writer = JsonlWriter(runs_path)
    runs_path.parent.mkdir(parents=True, exist_ok=True)

    completed: List[Dict[str, Any]] = []
    total_pairs = len(questions) * cfg.runs_per_question
    skipped = 0
    run_start = time.time()

    engine_info = engine.info()
    tqdm.write("model engine ready:")
    for key, value in engine_info.items():
        tqdm.write(f"  {key}: {value}")
    tqdm.write(
        f"experiment: {cfg.experiment_id}/{cfg.experiment_version} | "
        f"{len(questions)} questions x {cfg.runs_per_question} runs | "
        f"run_seeds={cfg.run_seeds}"
    )

    for question in questions:
        for run_index, run_seed in enumerate(cfg.run_seeds):
            run_id = f"{question['question_id']}-run-{run_index:02d}"
            if writer.contains(run_id):
                skipped += 1
                tqdm.write(f"skip existing run: {run_id}")
                continue
            started = time.time()
            record = orchestrator.run_one(
                question, run_index, run_seed, run_id
            )
            writer.append(record)
            completed.append(record)
            elapsed = time.time() - started
            tqdm.write(
                f"[{len(completed) + skipped}/{total_pairs}] {run_id} | "
                f"f1={record['f1']:.3f} em={record['em']:.3f} | "
                f"steps={record['decision_steps']} "
                f"gen_tok={record['total_generated_tokens']} "
                f"term={record['termination_reason']} | {elapsed:.1f}s"
            )

    total_elapsed = time.time() - run_start
    tqdm.write(
        f"done: {len(completed)} new runs, {skipped} skipped, "
        f"{total_elapsed:.1f}s total"
    )
    if completed and not skip_report:
        write_report(runs_path, runs_path.parent)
    return completed


def run_from_config(cfg: ExperimentConfig) -> None:
    """Load everything and run the experiment end to end."""
    tqdm.write(f"loading manifest: {cfg.manifest_path}")
    questions = [asdict(q) for q in load_manifest(cfg.manifest_path)]
    if cfg.num_questions and len(questions) > cfg.num_questions:
        questions = questions[: cfg.num_questions]
    if len(questions) < cfg.num_questions:
        raise RuntimeError(
            f"manifest has {len(questions)} questions but config requires "
            f"{cfg.num_questions}"
        )
    tqdm.write(f"selected {len(questions)} questions from manifest")

    prompts = PromptSet(cfg.prompt_dir)
    tqdm.write(f"prompts loaded, prompt_version={prompts.version}")
    environment_info = collect_environment_info()
    tqdm.write("environment collected")
    engine = build_engine(cfg)

    runs_path = cfg.output_subdir() / "runs.jsonl"
    run_experiment(cfg, questions, engine, environment_info, prompts, runs_path)
