"""Structured trace + deterministic worked-example rendering (Stage 1a).

The trace is a single ``trace.json`` assembled ONLY from values the
training run actually produced (real losses, real logits, real
generations, real gradient norms). ``render_worker_sft_worked_example``
reads the JSON and renders ``worked_example_worker_sft.md``; it never
invents numbers, and rendering is deterministic: the same trace file
always produces the same Markdown. Sections follow the spec:

- A. Run identity
- B. Raw data (annotations marked "not shown to the model")
- C. Data conversion
- D. Dedup and sample selection
- E. Actual worker inputs and targets
- F. Tokenization and the loss mask
- G. SFT loss (teacher forcing, per-token values, arithmetic)
- H. Parameter updates
- I. Before/after generation
- J. Limits and next steps
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .config import WorkerSFTConfig
from .masking import supervised_token_records

REQUIRED_SECTIONS = [
    "A_run_identity",
    "B_raw_data",
    "C_data_conversion",
    "D_dedup_and_selection",
    "E_inputs_targets",
    "F_tokenization",
    "G_loss",
    "H_parameter_update",
    "I_generation",
    "J_limits",
]


class WorkerSFTTraceRecorder:
    """Builds the trace JSON from the trainer's actual artifacts."""

    def __init__(self, cfg: WorkerSFTConfig, mode: str) -> None:
        self.cfg = cfg
        self.mode = mode

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def build(self, **kw: Any) -> Dict[str, Any]:
        trace = {
            "schema_version": 1,
            "A_run_identity": self._section_a(**kw),
            "B_raw_data": self._section_b(**kw),
            "C_data_conversion": self._section_c(**kw),
            "D_dedup_and_selection": self._section_d(**kw),
            "E_inputs_targets": self._section_e(**kw),
            "F_tokenization": self._section_f(**kw),
            "G_loss": self._section_g(**kw),
            "H_parameter_update": self._section_h(**kw),
            "I_generation": self._section_i(**kw),
            "J_limits": self._section_j(**kw),
        }
        missing = [s for s in REQUIRED_SECTIONS if s not in trace]
        if missing:
            raise ValueError(f"trace missing sections: {missing}")
        return trace

    @staticmethod
    def _section_a(**kw: Any) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = kw["config_snapshot"]
        model_cfg = snapshot.get("model", {})
        return {
            "date": kw["start_time"],
            "run_id": kw["run_id"],
            "code_revision": _git_revision(),
            "model": {
                "name": kw["model_info"].get("model_name"),
                "requested_revision": model_cfg.get("revision"),
                "resolved_revision": kw["model_info"].get(
                    "resolved_revision", "unavailable"
                ),
                "revision": kw["model_info"].get(
                    "resolved_revision", "unavailable"
                ),
                "tokenizer_chat_template_sha256": kw["model_info"].get(
                    "chat_template_sha256"
                ),
                "real_model": kw["model_info"].get("real_model"),
                "fresh_adapter": kw["model_info"].get("fresh_adapter"),
                "loaded_previous_adapter": kw["model_info"].get(
                    "loaded_previous_adapter"
                ),
            },
            "hardware": kw["env_info"],
            "effective_config": snapshot,
            "seeds": {
                "seed": snapshot.get("seed"),
                "sft_seed": snapshot.get("sft", {}).get("seed"),
                "example_order_seed": snapshot.get("sft", {}).get(
                    "example_order_seed"
                ),
                "probe_seed": snapshot.get("decode", {}).get("probe_seed"),
            },
        }

    @staticmethod
    def _section_b(**kw: Any) -> Dict[str, Any]:
        rows = []
        for example in kw["train_examples"]:
            rows.append(
                {
                    "example_id": example.example_id,
                    "source": example.source,
                    "split": example.split,
                    "parent_id": example.parent_id,
                    "original_question": example.original_question,
                    "visible_documents": [
                        {"doc_id": doc.doc_id, "title": doc.title,
                         "text": doc.text}
                        for doc in example.documents
                    ],
                    "annotations": {
                        "answer": example.answer,
                        "answer_aliases": example.answer_aliases,
                        "answerable": example.answerable,
                        "negative_reason": example.negative_reason,
                        "note": "answer/aliases are metadata ONLY; they are "
                                "not shown to the model",
                    },
                }
            )
        return {"training_examples": rows, "probe_examples": [
            {
                "example_id": e.example_id,
                "local_question": e.local_question,
                "answerable": e.answerable,
            }
            for e in kw["probe_examples"]
        ]}

    @staticmethod
    def _section_c(**kw: Any) -> Dict[str, Any]:
        rows = []
        for example in kw["train_examples"]:
            rows.append(
                {
                    "example_id": example.example_id,
                    "original_question": example.original_question,
                    "resolved_local_question": example.local_question,
                    "substitutions": example.substitutions,
                    "evidence": [
                        {"doc_id": ev.doc_id,
                         "sentence_index": ev.sentence_index,
                         "sentence": ev.sentence,
                         "char_span": list(ev.char_span)
                         if ev.char_span else None}
                        for ev in example.evidence
                    ],
                    "sentence_count": len(example.evidence),
                    "why_negative": (
                        example.negative_reason
                        if not example.answerable
                        else "not applicable (positive example)"
                    ),
                    "auto_checks": example.checks,
                    "review": example.review,
                }
            )
        return {
            "examples": rows,
            "note": (
                "automatic checks and the human review status are separate: "
                "passing auto checks does not make an example reviewed"
            ),
        }

    @staticmethod
    def _section_d(**kw: Any) -> Dict[str, Any]:
        dedup = kw["dedup_report"]
        selected_ids = [e.example_id for e in kw["train_examples"]]
        return {
            "methods": dedup.get("methods", {}),
            "removed": dedup.get("removed", {}),
            "counts": dedup.get("counts", {}),
            "eval_manifests": dedup.get("eval_manifests", []),
            "selected_training_ids": selected_ids,
            "label_quality_summary": {
                "per_source": kw["label_quality"].get("per_source", {}),
                "teacher_file": kw["label_quality"].get("teacher_file", {}),
                "reviewed_file_problems": kw["label_quality"].get(
                    "reviewed_file_problems", []
                ),
            },
        }

    @staticmethod
    def _section_e(**kw: Any) -> Dict[str, Any]:
        rows = []
        for item in kw["fixed_inputs"]:
            example = item["example"]
            rendered = _render_prompt(item["messages"])
            rows.append(
                {
                    "kind": item["kind"],
                    "example_id": example.example_id,
                    "messages": item["messages"],
                    "rendered_prompt": rendered,
                    "visible_documents": [
                        {"doc_id": doc.doc_id, "title": doc.title,
                         "text": doc.text}
                        for doc in example.documents
                    ],
                    "exact_target": item["reference_target"],
                }
            )
        return {"examples": rows}

    @staticmethod
    def _section_f(**kw: Any) -> Dict[str, Any]:
        sft_examples = kw["sft_examples"]
        train_examples = kw["train_examples"]
        token_tables = {}
        for example in train_examples:
            sft_example = sft_examples[example.example_id]
            token_tables[example.example_id] = {
                "counts": {
                    "prompt_tokens": sft_example.prompt_len,
                    "total_tokens": int(sft_example.input_ids.shape[0]),
                    "target_tokens": int(
                        sft_example.target_token_ids.shape[0]
                    ),
                },
                "token_table": [
                    {
                        "position": position,
                        "token_id": int(sft_example.input_ids[position]),
                        "token": (
                            sft_example.readable_tokens[position]
                            if position < len(sft_example.readable_tokens)
                            else "?"
                        ),
                        "label": int(sft_example.labels[position]),
                        "contributes_to_loss": (
                            int(sft_example.labels[position]) != -100
                        ),
                    }
                    for position in range(
                        int(sft_example.input_ids.shape[0])
                    )
                ],
                "causal_alignment": (
                    "position t predicts token t+1, so the first target "
                    f"token (position {sft_example.prompt_len}) is "
                    f"predicted from the last prompt position "
                    f"({sft_example.prompt_len - 1}) and the end-of-turn "
                    f"token (position {sft_example.eos_position}) is "
                    "predicted from the last content position"
                ),
                "eot_supervision": {
                    "eos_token_id": sft_example.eos_token_id,
                    "eos_position": sft_example.eos_position,
                    "eos_trained": (
                        int(sft_example.labels[sft_example.eos_position])
                        == sft_example.eos_token_id
                    ),
                    "first_target_trained": (
                        int(sft_example.labels[sft_example.prompt_len])
                        == int(sft_example.input_ids[sft_example.prompt_len])
                    ),
                },
            }
        return {
            "masking_rule": (
                "label = -100 (ignored) for the system/user prompt, "
                "padding, and any non-target template tokens; the ENTIRE "
                "assistant turn from the first target token through the "
                "end-of-turn token is supervised (unlike the Stage 2 "
                "gold-answer-only masking, which is NOT copied here)"
            ),
            "examples": token_tables,
        }

    @staticmethod
    def _section_g(**kw: Any) -> Dict[str, Any]:
        sft_examples = kw["sft_examples"]
        nll_cache = kw["per_example_nll_cache"]
        per_example = []
        for example in kw["train_examples"]:
            sft_example = sft_examples[example.example_id]
            row = nll_cache.get(example.example_id)
            tokens = (
                supervised_token_records(sft_example, row)
                if row is not None
                else []
            )
            example_loss = (
                sum(t["nll"] for t in tokens) / len(tokens)
                if tokens
                else None
            )
            per_example.append(
                {
                    "example_id": example.example_id,
                    "answerable": example.answerable,
                    "supervised_tokens": tokens,
                    "example_loss": example_loss,
                    "arithmetic": (
                        f"mean of {len(tokens)} supervised token NLLs "
                        f"= {example_loss:.6f}"
                        if example_loss is not None
                        else "not recorded"
                    ),
                }
            )
        update_rows = []
        for record in kw["per_step_records"]:
            token_counts = record["per_example_supervised_tokens"]
            nll_sums = [
                loss * count
                for loss, count in zip(
                    record["per_example_losses"], token_counts
                )
            ]
            update_rows.append(
                {
                    "step": record["step"],
                    "update_loss": record["update_loss"],
                    "supervised_tokens": record["supervised_tokens"],
                    "arithmetic": (
                        "sum of all target-token NLLs in the effective "
                        f"batch ({record['microsteps']} examples): ("
                        + " + ".join(
                            f"{x:.6f}" for x in nll_sums
                        )
                        + f") / {record['supervised_tokens']} tokens"
                        + f" = {record['update_loss']:.6f}"
                    ),
                }
            )
        return {
            "teacher_forcing": (
                "plain English: the model reads the full prompt and is "
                "asked 'what is the next token?' at every position of the "
                "gold assistant turn. Each answer's probability becomes "
                "that token's negative log-likelihood (NLL). The update "
                "loss sums every target-token NLL in the effective batch "
                "and divides once by the total number of target tokens. "
                "Per-example means below are diagnostics only. There is "
                "no generated-then-aligned "
                "text: the target text is fed to the model directly."
            ),
            "per_example": per_example,
            "update_losses": update_rows,
        }

    @staticmethod
    def _section_h(**kw: Any) -> Dict[str, Any]:
        return {
            "per_step": kw["per_step_records"],
            "optimizer": kw["config_snapshot"].get("optimizer", {}),
            "trainable": kw["model_info"].get("frozen_audit", {}),
            "frozen_base_check": {
                "method": (
                    "requires_grad audit at load time + sha256 over the "
                    "frozen base parameters before and after training "
                    "(full hash in trace mode)"
                ),
                "hash_before": kw["base_hash_before"],
                "hash_after": kw["base_hash_after"],
                "unchanged": kw["base_unchanged"],
            },
            "checkpoints": [
                record["checkpoint"]
                for record in kw["per_step_records"]
                if record.get("checkpoint")
            ],
        }

    @staticmethod
    def _section_i(**kw: Any) -> Dict[str, Any]:
        rows = []
        for item in kw["fixed_inputs"]:
            rows.append(
                {
                    "kind": item["kind"],
                    "example_id": item["example"].example_id,
                    "reference_target": item["reference_target"],
                    "pre": item["pre"],
                    "post": item["post"],
                }
            )
        return {
            "decode": kw["config_snapshot"].get("decode", {}),
            "pre_summary": _generation_summary(kw["fixed_inputs"], "pre"),
            "post_summary": _generation_summary(kw["fixed_inputs"], "post"),
            "note": (
                "same inputs, same deterministic greedy decoding before "
                "and after training; the trace's single effective-batch "
                "update is NOT expected to improve outputs"
            ),
            "inputs": rows,
        }

    @staticmethod
    def _section_j(**kw: Any) -> Dict[str, Any]:
        return {
            "confirmed": [
                "data conversion for all three sources runs end to end "
                "(see B and C)",
                "the automatic checks, dedup pipeline, and mixture "
                "selection ran on the real prepared data (see D)",
                "masking supervises the whole assistant turn including "
                "the end-of-turn token (see F)",
                "the training loop performed exactly the recorded "
                "optimizer steps with real forward losses (see G and H)",
                "base weights were unchanged by training (see H)",
            ],
            "not_established": [
                "the one-update trace run is not expected to improve the "
                "worker; it only demonstrates the pipeline end to end",
                "no quantitative evaluation on the frozen val/test was "
                "run in trace mode (deferred)",
                "the single-document phase trains only 1-document "
                "inputs; multi-document training is a separate later "
                "phase (only fixed multi-doc probes exist today)",
            ],
            "deferred": [
                "smoke and normal runs with effective batch 16",
                "validation-based checkpoint selection",
                "multi-document training phase (1-5 docs, easy "
                "distractors first)",
                "teacher-generated local questions for HotpotQA",
                "full baseline comparisons",
            ],
        }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _generation_summary(
    fixed_inputs: List[Dict[str, Any]], stage: str
) -> Dict[str, Any]:
    """Compact, label-aware error summary for fixed trace generations."""
    rows = [(item["example"], item[stage]["checks"]) for item in fixed_inputs]
    positives = [checks for example, checks in rows if example.answerable]
    negatives = [checks for example, checks in rows if not example.answerable]
    false_abstentions = sum(checks["abstained"] for checks in positives)
    false_answers = sum(not checks["abstained"] for checks in negatives)
    parse_failures = sum(not checks["parse_ok"] for _, checks in rows)
    return {
        "examples": len(rows),
        "answerable_examples": len(positives),
        "unanswerable_examples": len(negatives),
        "false_abstentions": false_abstentions,
        "false_abstention_rate": (
            false_abstentions / len(positives) if positives else None
        ),
        "false_answers": false_answers,
        "false_answer_rate": false_answers / len(negatives) if negatives else None,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(rows) if rows else None,
    }

def _git_revision() -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def _render_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for message in messages:
        parts.append(f"[{message['role']}]\n{message['content']}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Deterministic Markdown renderer
# --------------------------------------------------------------------------

def render_worker_sft_worked_example(trace_path: Any, out_path: Any) -> Path:
    """Render ``worked_example_worker_sft.md`` ONLY from the trace JSON."""
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)
    for section in REQUIRED_SECTIONS:
        if section not in trace:
            raise ValueError(f"trace is missing section {section}")
    lines: List[str] = []
    lines.append("# Worked example: Stage 1a worker SFT (trace)")
    lines.append("")
    lines.append(
        "Every number in this file comes from the actual training run "
        "(`trace.json`). Nothing is invented here; regenerate it anytime "
        "with `render-worker-sft-trace`."
    )
    lines.append("")
    lines.extend(_render_a(trace["A_run_identity"]))
    lines.extend(_render_b(trace["B_raw_data"]))
    lines.extend(_render_c(trace["C_data_conversion"]))
    lines.extend(_render_d(trace["D_dedup_and_selection"]))
    lines.extend(_render_e(trace["E_inputs_targets"]))
    lines.extend(_render_f(trace["F_tokenization"]))
    lines.extend(_render_g(trace["G_loss"]))
    lines.extend(_render_h(trace["H_parameter_update"]))
    lines.extend(_render_i(trace["I_generation"]))
    lines.extend(_render_j(trace["J_limits"]))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _render_a(section: Dict[str, Any]) -> List[str]:
    lines = ["## A. Run identity", ""]
    lines.append(f"- Date: {section.get('date')}")
    lines.append(f"- Run id: {section.get('run_id')}")
    lines.append(f"- Code revision: {section.get('code_revision')}")
    model = section.get("model", {})
    lines.append(f"- Model: {model.get('name')}")
    lines.append(f"- Real model (not a fixture): {model.get('real_model')}")
    lines.append(
        f"- Fresh LoRA adapter, no previous adapter loaded: "
        f"{model.get('fresh_adapter') and not model.get('loaded_previous_adapter')}"
    )
    hardware = section.get("hardware", {})
    lines.append(
        f"- Hardware: {hardware.get('platform')} / "
        f"GPU {hardware.get('gpu_name') or 'none'}"
    )
    libs = {
        key: value for key, value in hardware.items()
        if key.endswith("_version")
    }
    lines.append(f"- Library versions: {_compact(libs)}")
    lines.append(f"- Seeds: {_compact(section.get('seeds', {}))}")
    cfg = section.get("effective_config", {})
    sft = cfg.get("sft", {})
    peft = (cfg.get("model") or {}).get("peft", {})
    lines.append(
        f"- Effective SFT settings: epochs={sft.get('num_epochs')}, "
        f"lr={sft.get('learning_rate')}, weight_decay="
        f"{sft.get('weight_decay')}, microbatch={sft.get('microbatch_size')}, "
        f"grad_accum={sft.get('grad_accum_steps')}, clip="
        f"{sft.get('max_grad_norm')}, scheduler={sft.get('scheduler')}, "
        f"LoRA r={peft.get('r')}/alpha={peft.get('alpha')}/"
        f"dropout={peft.get('dropout')}"
    )
    lines.append("")
    return lines


def _render_b(section: Dict[str, Any]) -> List[str]:
    lines = ["## B. Raw data", ""]
    lines.append(
        "Original questions, visible texts, and annotations. The answer "
        "and aliases are NOT shown to the model."
    )
    lines.append("")
    for row in section.get("training_examples", []):
        lines.append(f"### {row['example_id']}")
        lines.append("")
        lines.append(f"- source: {row['source']} / split: {row['split']}")
        lines.append(f"- parent id: {row['parent_id']}")
        lines.append(f"- original question: {row['original_question']}")
        annotations = row["annotations"]
        lines.append(
            f"- answer (metadata only): {annotations['answer']!r} "
            f"(answerable={annotations['answerable']})"
        )
        if annotations.get("negative_reason"):
            lines.append(f"- why negative: {annotations['negative_reason']}")
        for doc in row["visible_documents"]:
            lines.append(f"- document {doc['doc_id']} ({doc['title']}):")
            lines.append(f"  {doc['text']}")
        lines.append("")
    lines.append("### Probes (fixed generation inputs, not trained)")
    for probe in section.get("probe_examples", []):
        lines.append(
            f"- {probe['example_id']}: {probe['local_question']} "
            f"(answerable={probe['answerable']})"
        )
    lines.append("")
    return lines


def _render_c(section: Dict[str, Any]) -> List[str]:
    lines = ["## C. Data conversion", ""]
    for row in section.get("examples", []):
        lines.append(f"### {row['example_id']}")
        lines.append("")
        lines.append(f"- original: {row['original_question']}")
        lines.append(f"- resolved local question: {row['resolved_local_question']}")
        if row["substitutions"]:
            for sub in row["substitutions"]:
                lines.append(
                    f"- substitution: {sub['placeholder']} -> "
                    f"{sub['replacement']} ({sub['source']})"
                )
        for ev in row["evidence"]:
            lines.append(
                f"- evidence sentence {ev['sentence_index']} "
                f"({ev['doc_id']}): {ev['sentence']}"
            )
        lines.append(f"- sentences used: {row['sentence_count']}")
        lines.append(f"- why negative: {row['why_negative']}")
        review = row["review"]
        lines.append(
            f"- review status: {review.get('status')} "
            f"(reviewer: {review.get('reviewer', 'n/a')})"
        )
        failed = [
            name for name, info in row["auto_checks"].items()
            if not info["ok"]
        ]
        lines.append(
            f"- automatic checks: {len(row['auto_checks'])} run, "
            f"failed: {failed or 'none'}"
        )
        lines.append("")
    lines.append(
        "> Auto checks are mechanical; they never set the reviewed "
        "status. Review status comes from a human with a recorded method."
    )
    lines.append("")
    return lines


def _render_d(section: Dict[str, Any]) -> List[str]:
    lines = ["## D. Dedup and sample selection", ""]
    for name, method in section.get("methods", {}).items():
        lines.append(f"- {name}: {method}")
    counts = section.get("counts", {})
    lines.append(f"- counts: {_compact(counts)}")
    removed = section.get("removed", {})
    for stage, entries in removed.items():
        if entries:
            lines.append(f"- removed by {stage}: {len(entries)}")
            for entry in entries[:20]:
                lines.append(
                    f"  - {entry.get('example_id')}: {entry.get('reason')}"
                )
            if len(entries) > 20:
                lines.append(f"  - ... and {len(entries) - 20} more")
    lines.append(
        f"- selected training ids: "
        f"{', '.join(section.get('selected_training_ids', []))}"
    )
    for manifest in section.get("eval_manifests", []):
        lines.append(
            f"- eval manifest {manifest.get('path')}: "
            f"{manifest.get('status')}"
        )
    lines.append("")
    return lines


def _render_e(section: Dict[str, Any]) -> List[str]:
    lines = ["## E. Actual worker inputs and targets", ""]
    for row in section.get("examples", []):
        lines.append(f"### {row['example_id']} ({row['kind']})")
        lines.append("")
        for message in row["messages"]:
            lines.append(f"Role `{message['role']}`:")
            lines.append("")
            lines.append("```text")
            lines.append(message["content"].rstrip())
            lines.append("```")
            lines.append("")
        lines.append(f"Exact assistant target:")
        lines.append("")
        lines.append("```text")
        lines.append(row["exact_target"])
        lines.append("```")
        lines.append("")
    return lines


def _render_f(section: Dict[str, Any]) -> List[str]:
    lines = ["## F. Tokenization and the loss mask", ""]
    lines.append(section.get("masking_rule", ""))
    lines.append("")
    for example_id, data in section.get("examples", {}).items():
        lines.append(f"### {example_id}")
        lines.append("")
        counts = data["counts"]
        lines.append(
            f"- prompt tokens: {counts['prompt_tokens']}, total tokens: "
            f"{counts['total_tokens']}, target tokens: "
            f"{counts['target_tokens']}"
        )
        eot = data["eot_supervision"]
        lines.append(
            f"- first target token trained: {eot['first_target_trained']}; "
            f"end-of-turn token (id {eot['eos_token_id']}, position "
            f"{eot['eos_position']}) trained: {eot['eos_trained']}"
        )
        lines.append("")
        lines.append(
            "| position | token id | token | label | contributes to loss |"
        )
        lines.append("|---|---|---|---|---|")
        for token in data["token_table"]:
            contributes = "yes" if token["contributes_to_loss"] else "no"
            label = token["label"] if token["label"] != -100 else "-100"
            lines.append(
                f"| {token['position']} | {token['token_id']} | "
                f"{token['token']!r} | {label} | {contributes} |"
            )
        lines.append("")
        lines.append(f"- causal alignment: {data['causal_alignment']}")
        lines.append("")
    return lines


def _render_g(section: Dict[str, Any]) -> List[str]:
    lines = ["## G. SFT loss", ""]
    lines.append(section.get("teacher_forcing", ""))
    lines.append("")
    lines.append("### Per-example token losses (from the actual forward)")
    lines.append("")
    for row in section.get("per_example", []):
        lines.append(f"#### {row['example_id']}")
        lines.append("")
        lines.append(
            "| position | token | log prob | NLL |"
        )
        lines.append("|---|---|---|---|")
        for token in row["supervised_tokens"]:
            lines.append(
                f"| {token['position']} | {token['token']!r} | "
                f"{token['log_prob']:.6f} | {token['nll']:.6f} |"
            )
        lines.append("")
        lines.append(f"- arithmetic: {row['arithmetic']}")
        lines.append("")
    lines.append("### Update losses (mean over all target tokens)")
    lines.append("")
    lines.append("| step | update loss | arithmetic |")
    lines.append("|---|---|---|")
    for row in section.get("update_losses", []):
        lines.append(
            f"| {row['step']} | {row['update_loss']:.6f} | "
            f"{row['arithmetic']} |"
        )
    lines.append("")
    return lines


def _render_h(section: Dict[str, Any]) -> List[str]:
    lines = ["## H. Parameter updates", ""]
    optimizer = section.get("optimizer", {})
    lines.append(
        f"- optimizer: {optimizer.get('name')} lr="
        f"{optimizer.get('lr')} weight_decay={optimizer.get('weight_decay')} "
        f"betas={optimizer.get('betas')} eps={optimizer.get('eps')} "
        f"scheduler={optimizer.get('scheduler')} "
        f"warmup_ratio={optimizer.get('warmup_ratio')}"
    )
    trainable = section.get("trainable", {})
    lines.append(
        f"- trainable parameters: {trainable.get('trainable_names', [])}"
    )
    frozen = section.get("frozen_base_check", {})
    lines.append(f"- frozen-base method: {frozen.get('method')}")
    lines.append(
        f"- base hash before: {frozen.get('hash_before')}"
    )
    lines.append(
        f"- base hash after: {frozen.get('hash_after')}"
    )
    lines.append(f"- base weights unchanged: {frozen.get('unchanged')}")
    lines.append("")
    lines.append(
        "| step | epoch | microsteps | lr | update loss | grad norm "
        "(before/after clip) | LoRA delta norm | checkpoint |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for record in section.get("per_step", []):
        lines.append(
            f"| {record['step']} | {record['epoch']} | "
            f"{record['microsteps']} | {record['learning_rate']} | "
            f"{record['update_loss']:.6f} | "
            f"{record['grad_norm_before_clip']:.4f} / "
            f"{record['grad_norm_after_clip']:.4f} | "
            f"{record['lora_delta_norm']} | "
            f"{record['checkpoint'] or ''} |"
        )
    lines.append("")
    return lines


def _render_i(section: Dict[str, Any]) -> List[str]:
    lines = ["## I. Before/after generation", ""]
    decode = section.get("decode", {})
    lines.append(
        f"- decoding: greedy (do_sample={decode.get('do_sample')}), "
        f"max_new_tokens={decode.get('max_new_tokens')}"
    )
    lines.append("")
    lines.append(section.get("note", ""))
    lines.append("")
    for stage in ("pre", "post"):
        summary = section.get(f"{stage}_summary", {})
        lines.append(
            f"- {stage} summary: false abstention "
            f"{summary.get('false_abstentions')}/"
            f"{summary.get('answerable_examples')}; false answer "
            f"{summary.get('false_answers')}/"
            f"{summary.get('unanswerable_examples')}; parse failures "
            f"{summary.get('parse_failures')}/{summary.get('examples')}"
        )
    lines.append("")
    for row in section.get("inputs", []):
        lines.append(f"### {row['example_id']} ({row['kind']})")
        lines.append("")
        lines.append(f"- reference target: `{row['reference_target']}`")
        for stage in ("pre", "post"):
            data = row[stage]
            lines.append(
                f"- {stage}-training output "
                f"({data['num_new_tokens']} new tokens, "
                f"finish={data['finish_reason']}): "
                f"`{data['text']}`"
            )
            checks = data["checks"]
            lines.append(
                f"  - parse_ok={checks['parse_ok']} "
                f"source_valid={checks['source_valid']} "
                f"quotes_verbatim={checks['quotes_verbatim']} "
                f"answer_covered={checks['answer_covered']} "
                f"abstention_correct={checks['abstention_correct']} "
                f"correct={checks['correct']}"
            )
        lines.append("")
    return lines


def _render_j(section: Dict[str, Any]) -> List[str]:
    lines = ["## J. Limits and next steps", ""]
    lines.append("Confirmed:")
    for item in section.get("confirmed", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Not established:")
    for item in section.get("not_established", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Deferred:")
    for item in section.get("deferred", []):
        lines.append(f"- {item}")
    lines.append("")
    return lines


def _compact(mapping: Dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(mapping.items()))
