"""Fail-fast learner checks that run before any expensive training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from hotpot_mas.question_selection import load_manifest
from hotpot_mas.seeds import derive_generation_seed, seed_all

from .config import TrainingConfig
from .data import manifest_path_for_split
from .policy import HFPolicy
from .prompts_builder import TrainingPrompts


def check_logprob_consistency(
    cfg: TrainingConfig,
    mean_tolerance: float,
    max_tolerance: float,
) -> Dict[str, Any]:
    """Compare sampling-time and immediate teacher-forced token log-probs."""
    if mean_tolerance < 0 or max_tolerance < 0:
        raise ValueError("log-prob tolerances must be non-negative")
    questions = load_manifest(manifest_path_for_split(cfg, "train"))
    if not questions:
        raise RuntimeError("training manifest contains no questions")
    question = questions[0]
    seed_all(cfg.seed)
    policy = HFPolicy(cfg.model)
    prompts = TrainingPrompts(cfg.prompt_dir)
    side_results: List[Dict[str, Any]] = []
    all_diffs: List[torch.Tensor] = []
    for side, evidence in (
        ("A", question.evidence_alice),
        ("B", question.evidence_bob),
    ):
        messages = prompts.worker_messages(side, question.question, evidence)
        seed = derive_generation_seed(
            cfg.seed, f"logprob-check:{question.question_id}:{side}", 0
        )
        report = policy.sample_report(messages, seed, cfg.workers.rollout)
        if not report.token_ids:
            raise RuntimeError(f"log-prob check produced an empty {side} report")
        with torch.no_grad():
            recomputed = policy.teacher_force(
                policy.tokenize(messages),
                report.token_ids,
                cfg.workers.rollout,
            ).detach().float().cpu()
        sampled = torch.tensor(report.logprobs, dtype=torch.float32)
        if sampled.shape != recomputed.shape:
            raise RuntimeError(
                f"{side} log-prob shape mismatch: sampled={sampled.shape}, "
                f"recomputed={recomputed.shape}"
            )
        diff = (sampled - recomputed).abs()
        all_diffs.append(diff)
        side_results.append(
            {
                "side": side,
                "num_tokens": len(report.token_ids),
                "finish_reason": report.finish_reason,
                "mean_absolute_difference": float(diff.mean().item()),
                "max_absolute_difference": float(diff.max().item()),
                "sampled_logprob_mean": float(sampled.mean().item()),
                "recomputed_logprob_mean": float(recomputed.mean().item()),
            }
        )
    combined = torch.cat(all_diffs)
    mean_diff = float(combined.mean().item())
    max_diff = float(combined.max().item())
    passed = mean_diff <= mean_tolerance and max_diff <= max_tolerance
    result = {
        "question_id": question.question_id,
        "model": cfg.model.to_dict(),
        "decode": cfg.workers.rollout.to_dict(),
        "mean_tolerance": mean_tolerance,
        "max_tolerance": max_tolerance,
        "mean_absolute_difference": mean_diff,
        "max_absolute_difference": max_diff,
        "sides": side_results,
        "passed": passed,
    }
    output_path = cfg.stage_dir() / "logprob_consistency.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    result["output_path"] = str(output_path)
    if not passed:
        raise RuntimeError(
            "log-prob consistency gate failed: "
            f"mean_abs_diff={mean_diff:.6g} (tol={mean_tolerance:.6g}), "
            f"max_abs_diff={max_diff:.6g} (tol={max_tolerance:.6g}); "
            f"details: {output_path}"
        )
    return result
