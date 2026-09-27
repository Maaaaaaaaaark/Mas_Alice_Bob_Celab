"""Trajectory loop: load config + manifest, run questions x seeds, log.

Runs are written to ``runs.jsonl`` one line at a time (append mode) so an
interrupted run can be resumed: existing runs are skipped only when their
trajectory fingerprints match the current inputs and resolved engine.
Each run's progress is printed to stdout via ``tqdm.write``. The report is
written automatically at the end of a non-empty run set.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from .config import ExperimentConfig
from .logging_io import JsonlWriter, collect_environment_info
from .model_engine import HFEngine, ModelEngine, VLLMEngine
from .orchestrator import Orchestrator
from .prompts import (
    CENTRALIZED_PROMPT_NAMES,
    MAS_PROMPT_NAMES,
    ONE_SHOT_PROMPT_NAMES,
    PromptSet,
)
from .question_selection import load_manifest
from .report import write_report


def run_fingerprint(
    cfg: ExperimentConfig,
    question: Dict[str, Any],
    run_index: int,
    run_seed: int,
    prompts: PromptSet,
    engine_info: Dict[str, Any],
    environment_info: Dict[str, Any],
) -> str:
    """Hash every input that can change one trajectory's model outputs."""
    payload = {
        "experiment_id": cfg.experiment_id,
        "experiment_version": cfg.experiment_version,
        "dataset": cfg.dataset,
        "dataset_config": cfg.dataset_config,
        "dataset_split": cfg.dataset_split,
        "sample_selection_seed": cfg.sample_selection_seed,
        "question": question,
        "run_index": run_index,
        "run_seed": run_seed,
        "model_name": cfg.model_name,
        "engine_info": engine_info,
        "runtime_environment": {
            key: environment_info.get(key)
            for key in (
                "python_version",
                "numpy_version",
                "torch_version",
                "cuda_version",
                "gpu_name",
                "transformers_version",
                "datasets_version",
            )
        },
        "dtype": cfg.dtype,
        "attn_implementation": cfg.attn_implementation,
        "max_input_length": cfg.max_input_length,
        "generation": cfg.generation.to_dict(),
        "max_decision_steps": cfg.max_decision_steps,
        "prompt_hashes": prompts.hashes,
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_engine(cfg: ExperimentConfig) -> ModelEngine:
    """Construct the shared frozen model engine (one instance per process)."""
    if cfg.engine == "vllm":
        return VLLMEngine(
            model_name=cfg.model_name,
            generation_params=cfg.generation,
            dtype=cfg.dtype,
            model_revision=cfg.model_revision,
            tokenizer_revision=cfg.tokenizer_revision,
            max_input_length=cfg.max_input_length,
            gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            enforce_eager=cfg.vllm_enforce_eager,
        )
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
    engine: ModelEngine,
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
    cfg.validate()
    total_pairs = len(questions) * len(cfg.run_seeds)
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
            run_id = f"{question['question_id']}-seed-{run_seed}"
            fingerprint = run_fingerprint(
                cfg,
                question,
                run_index,
                run_seed,
                prompts,
                engine_info,
                environment_info,
            )
            if writer.contains(run_id):
                existing = writer.get(run_id) or {}
                existing_fingerprint = existing.get("run_fingerprint")
                if existing_fingerprint != fingerprint:
                    raise RuntimeError(
                        f"existing run {run_id} has a different or missing "
                        "run_fingerprint; use a new experiment_version/output "
                        "directory instead of mixing configurations"
                    )
                skipped += 1
                tqdm.write(f"skip existing run: {run_id}")
                continue
            started = time.time()
            record = orchestrator.run_one(
                question, run_index, run_seed, run_id
            )
            record["run_fingerprint"] = fingerprint
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
    if writer.existing_ids and not skip_report:
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

    if cfg.architecture == "centralized_reader":
        prompt_names = CENTRALIZED_PROMPT_NAMES
    elif cfg.architecture in {
        "one_shot_gather",
        "one_shot_direct_answer",
    }:
        prompt_names = ONE_SHOT_PROMPT_NAMES
    else:
        prompt_names = MAS_PROMPT_NAMES
    prompts = PromptSet(cfg.prompt_dir, names=prompt_names)
    tqdm.write(f"prompts loaded, prompt_version={prompts.version}")
    environment_info = collect_environment_info()
    tqdm.write("environment collected")
    engine = build_engine(cfg)

    runs_path = cfg.output_subdir() / "runs.jsonl"
    run_experiment(cfg, questions, engine, environment_info, prompts, runs_path)
