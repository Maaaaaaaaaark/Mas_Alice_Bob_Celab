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
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from .report import write_report
from .sharding import (
    load_questions_for_config,
    select_question_shard,
    shard_output_dir,
)


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


def build_engine(
    cfg: ExperimentConfig, device: str | None = None
) -> ModelEngine:
    """Construct one frozen model engine."""
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
        device=device or cfg.device,
        attn_implementation=cfg.attn_implementation,
        model_revision=cfg.model_revision,
        tokenizer_revision=cfg.tokenizer_revision,
        max_input_length=cfg.max_input_length,
    )


def build_engines(cfg: ExperimentConfig) -> ModelEngine | Dict[str, ModelEngine]:
    """Build one shared engine or three physically independent model copies."""
    if cfg.model_instance_mode == "shared":
        return build_engine(cfg)
    return {
        name: build_engine(cfg, device=cfg.agent_devices[name])
        for name in ("alice", "bob", "celab")
    }


def engine_metadata(
    engine: ModelEngine | Dict[str, ModelEngine]
) -> Dict[str, Any]:
    """Describe model-instance topology for logs and trajectory hashes."""
    if not isinstance(engine, dict):
        # Preserve the legacy payload so old shared-engine runs can still be
        # resumed without a false fingerprint mismatch.
        return dict(engine.info())
    by_agent = {
        name: dict(agent_engine.info())
        for name, agent_engine in sorted(engine.items())
    }
    info = dict(by_agent.get("celab", next(iter(by_agent.values()))))
    info.update(
        {
            "model_instance_mode": "independent",
            "physical_model_instances": len(
                {id(agent_engine) for agent_engine in engine.values()}
            ),
            "by_agent": by_agent,
        }
    )
    return info


def run_experiment(
    cfg: ExperimentConfig,
    questions: List[Any],
    engine: ModelEngine | Dict[str, ModelEngine],
    environment_info: Dict[str, Any],
    prompts: PromptSet,
    runs_path: Path,
    skip_report: bool = False,
    shard_metadata: Optional[Dict[str, int]] = None,
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

    engine_info = engine_metadata(engine)
    tqdm.write("model engine(s) ready:")
    if isinstance(engine, dict):
        tqdm.write(
            "  topology: three independent physical model instances"
        )
        for name, info in engine_info["by_agent"].items():
            tqdm.write(
                f"  {name}: {info.get('model_name')} on "
                f"{info.get('device')} ({info.get('dtype')})"
            )
    else:
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
            if shard_metadata is not None:
                record["shard"] = dict(shard_metadata)
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


def run_from_config(
    cfg: ExperimentConfig,
    num_shards: int = 1,
    shard_index: int = 0,
) -> None:
    """Load everything and run the experiment end to end."""
    tqdm.write(f"loading manifest: {cfg.manifest_path}")
    all_questions = load_questions_for_config(cfg)
    questions = select_question_shard(
        all_questions, num_shards, shard_index
    )
    if not questions:
        raise RuntimeError(
            f"shard {shard_index}/{num_shards} contains no questions; "
            "use fewer shards"
        )
    tqdm.write(
        f"selected shard {shard_index}/{num_shards}: {len(questions)} of "
        f"{len(all_questions)} configured questions"
    )

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
    engine = build_engines(cfg)

    if num_shards == 1:
        runs_path = cfg.output_subdir() / "runs.jsonl"
        shard_metadata = None
    else:
        runs_path = shard_output_dir(
            cfg, num_shards, shard_index
        ) / "runs.jsonl"
        shard_metadata = {
            "num_shards": num_shards,
            "shard_index": shard_index,
        }
    run_experiment(
        cfg,
        questions,
        engine,
        environment_info,
        prompts,
        runs_path,
        shard_metadata=shard_metadata,
    )
