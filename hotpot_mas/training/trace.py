"""Trace recording and the auto-generated ``worked_example.md``.

``TraceRecorder`` accumulates the exhaustive per-question trace during a
trace-mode run; ``write_trace_json`` validates that every required field is
present before writing. ``render_worked_example`` reads **only** the
trace.json file and renders the human-readable English walkthrough — every
number in the markdown comes from the recorded run, never from a template
or hand-fabrication. Re-rendering the same trace.json is deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from hotpot_mas.logging_io import collect_environment_info

REQUIRED_KEYS = [
    "schema_version",
    "run",
    "dataset",
    "question",
    "documents",
    "partition",
    "prompts",
    "reports",
    "pairs",
    "reward_matrix",
    "marginals",
    "advantages",
    "kept",
    "loss",
    "diagnostics",
    "parameters",
    "checkpoint",
    "environment",
]

REQUIRED_SUBKEYS: Dict[str, List[str]] = {
    "run": ["mode", "config_summary"],
    "dataset": ["dataset", "dataset_config", "split", "question_id", "seed"],
    "question": ["question", "gold"],
    "partition": [
        "rule",
        "evidence_alice",
        "evidence_bob",
        "alice_documents",
        "bob_documents",
    ],
    "prompts": [
        "worker_a_messages",
        "worker_b_messages",
        "prompt_ids_length_a",
        "prompt_ids_length_b",
        "decode_params",
        "prompt_version",
    ],
    "reports": ["A", "B"],
    "marginals": ["q_a", "q_b", "std_a", "std_b", "signal_a", "signal_b"],
    "advantages": ["a", "b"],
    "kept": ["sides", "reports", "excluded_empty_reports"],
    "loss": ["per_report", "J", "L"],
    "diagnostics": [
        "approx_kl",
        "clip_fraction",
        "entropy",
        "grad_norm",
        "learning_rate",
        "num_policy_epochs",
    ],
    "parameters": ["before", "after"],
    "checkpoint": ["path", "saved"],
}


class TraceRecorder:
    """Thin validated accumulator for one trace-mode run."""

    def __init__(self, **initial: Any):
        self.data: Dict[str, Any] = {
            "schema_version": 1,
            **initial,
        }

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def update(self, mapping: Dict[str, Any]) -> None:
        self.data.update(mapping)

    def to_dict(self) -> Dict[str, Any]:
        return self.data


def _validate_trace(data: Dict[str, Any]) -> List[str]:
    """Return the list of missing required keys (empty when complete)."""
    missing = [key for key in REQUIRED_KEYS if key not in data]
    for key, subkeys in REQUIRED_SUBKEYS.items():
        if key in data:
            for sub in subkeys:
                if sub not in data[key]:
                    missing.append(f"{key}.{sub}")
    return missing


def write_trace_json(trace_path: Path, data: Dict[str, Any]) -> Path:
    """Validate and write the trace; raises on missing required fields."""
    missing = _validate_trace(data)
    if missing:
        raise ValueError(
            "trace missing required fields: " + ", ".join(sorted(missing))
        )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return trace_path


def load_trace_json(trace_path: Path) -> Dict[str, Any]:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    missing = _validate_trace(data)
    if missing:
        raise ValueError(
            f"{trace_path} is not a complete trace; missing: "
            + ", ".join(sorted(missing))
        )
    return data


# --------------------------------------------------------------------------
# worked_example.md renderer (reads trace.json only)
# --------------------------------------------------------------------------

def _f(value: Any, digits: int = 4) -> str:
    """Format a float with fixed digits; pass through non-floats."""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_f(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_worked_example(
    trace_path: Path, out_path: Path
) -> str:
    """Render the markdown walkthrough from a trace.json (no other input)."""
    t = load_trace_json(trace_path)
    md: List[str] = []
    add = md.append

    add("# Cross-Paired GRPO on One HotpotQA Question — Worked Example")
    add("")
    add("_Every number below comes from the recorded trace run "
        f"(`{trace_path.name}`); nothing is hand-fabricated._")
    add("")

    # 1. Original example -------------------------------------------------
    ds = t["dataset"]
    q = t["question"]
    add("## 1. Original HotpotQA example")
    add("")
    add(f"- dataset: `{ds['dataset']}` (config `{ds['dataset_config']}`), "
        f"split `{ds['split']}`")
    add(f"- question id: `{ds['question_id']}`")
    add(f"- run seed: `{ds['seed']}`")
    add(f"- **Question:** {q['question']}")
    add(f"- **Gold answer:** {q['gold']}")
    add("")

    # 2. Official fields vs preprocessing ---------------------------------
    add("## 2. Official fields vs preprocessing")
    add("")
    add("The official distractor context holds 10 documents. The pipeline "
        "keeps questions with a non-empty gold answer, type bridge/comparison, "
        "exactly 2 supporting titles, and 10 distinct non-empty documents. "
        "The table lists each document with its **gold supporting label** — "
        "these labels are used only for the oracle-balanced partition and "
        "never appear in any model prompt.")
    add("")
    rows = []
    for k, doc in enumerate(t["documents"]):
        rows.append([
            k,
            doc["title"],
            "supporting" if doc["is_supporting"] else "distractor",
        ])
    add(_md_table(["#", "Title", "Gold label"], rows))
    add("")

    # 3. Document partition ----------------------------------------------
    add("## 3. Document partition")
    add("")
    p = t["partition"]
    add(f"Partition rule: `{p['rule']}`. Alice and Bob each receive one "
        "supporting document and four distractors; the within-worker order "
        "is shuffled deterministically per question (fixed seed).")
    add("")
    add("**Alice's document order:**")
    add("")
    add(_md_table(["#", "Title", "Gold label"],
                  [[k, d["title"], d["is_supporting"]]
                   for k, d in enumerate(p["alice_documents"])]))
    add("")
    add("**Bob's document order:**")
    add("")
    add(_md_table(["#", "Title", "Gold label"],
                  [[k, d["title"], d["is_supporting"]]
                   for k, d in enumerate(p["bob_documents"])]))
    add("")
    add("The models see only the label-free evidence blocks below.")
    add("")
    add("### Alice's private evidence (as given to the model)")
    add("")
    add("```")
    add(p["evidence_alice"])
    add("```")
    add("")
    add("### Bob's private evidence (as given to the model)")
    add("")
    add("```")
    add(p["evidence_bob"])
    add("```")
    add("")

    # 4. Actual prompts ----------------------------------------------------
    add("## 4. Actual A/B prompts")
    add("")
    pr = t["prompts"]
    for name, key in (("Alice", "worker_a_messages"), ("Bob", "worker_b_messages")):
        add(f"### {name}")
        add("")
        for message in pr[key]:
            add(f"**role `{message['role']}`:**")
            add("")
            add("```")
            add(message["content"])
            add("```")
            add("")
    add(f"- tokenized prompt length (Alice): {pr['prompt_ids_length_a']}")
    add(f"- tokenized prompt length (Bob): {pr['prompt_ids_length_b']}")
    add(f"- prompt version: `{pr['prompt_version']}`")
    add(f"- decode parameters: `{json.dumps(pr['decode_params'], sort_keys=True)}`")
    add("")

    # 5. Sampled reports ----------------------------------------------------
    add("## 5. Sampled reports (old policy)")
    add("")
    reports = t["reports"]
    for side, label in (("A", "Alice"), ("B", "Bob")):
        add(f"### {label} (G = {len(reports[side])})")
        add("")
        rows = []
        for r in reports[side]:
            text = r["text"].replace("\n", " ")
            if len(text) > 300:
                text = text[:300] + "…"
            rows.append([
                r["index"],
                r["num_tokens"],
                r["finish_reason"],
                text,
            ])
        add(_md_table(["index", "tokens", "finish", "text (truncated)"], rows))
        add("")
    add("Token ids and old-policy log probs per report:")
    add("")
    for side, label in (("A", "Alice"), ("B", "Bob")):
        add(f"```")
        for r in reports[side]:
            add(f"{label}[{r['index']}] ids={r['token_ids']}")
            add(f"{label}[{r['index']}] logp={[_f(v) for v in r['logprobs']]}")
        add("```")
    add("")

    # 6. Cross-paired C answers --------------------------------------------
    add("## 6. All cross-paired C answers")
    add("")
    rows = []
    for row in t["pairs"]:
        for pair in row:
            rows.append([
                f"A{pair['i']} × B{pair['j']}",
                pair["pred_answer"],
                pair["f1"],
                pair["generated_tokens"],
                pair["parsed"],
            ])
    add(_md_table(["pair", "C's parsed answer", "F1", "C tokens", "tag parsed"],
                  rows))
    add("")

    # 7. Reward matrix ------------------------------------------------------
    add("## 7. Reward matrix")
    add("")
    add("Rows are Alice's report index i, columns Bob's report index j; "
        "R[i][j] = F1(C(q, a_i, b_j), y*).")
    add("")
    rows = [[f"a{i}"] + [_f(v) for v in row] for i, row in enumerate(t["reward_matrix"])]
    header = [""] + [f"b{j}" for j in range(len(t["reward_matrix"]))]
    add(_md_table(header, rows))
    add("")

    # 8. Marginals and advantages -------------------------------------------
    add("## 8. Marginal rewards and advantages")
    add("")
    m = t["marginals"]
    add(f"- Q_A = {[_f(v) for v in m['q_a']]}  (std {_f(m['std_a'])}, "
        f"signal: **{m['signal_a']}**)")
    add(f"- Q_B = {[_f(v) for v in m['q_b']]}  (std {_f(m['std_b'])}, "
        f"signal: **{m['signal_b']}**)")
    add("")
    if m["signal_a"]:
        add(f"- normalized advantages A: "
            f"{[_f(v) for v in t['advantages']['a']]}")
    if m["signal_b"]:
        add(f"- normalized advantages B: "
            f"{[_f(v) for v in t['advantages']['b']]}")
    add("")
    k = t["kept"]
    add(f"- kept sides: {k['sides']}; kept reports S_q: {k['reports']}")
    if k["excluded_empty_reports"]:
        add(f"- empty reports excluded from the loss: "
            f"{k['excluded_empty_reports']}")
    add("")

    # 9. GRPO loss construction ---------------------------------------------
    add("## 9. GRPO loss construction")
    add("")
    rows = []
    for entry in t["loss"]["per_report"]:
        ratios = entry["ratio"]
        rows.append([
            f"{entry['side']}{entry['index']}",
            entry["tokens"],
            entry["weight"],
            _f(entry["objective"]),
            _f(sum(ratios) / len(ratios)) if ratios else "-",
        ])
    add(_md_table(["report", "tokens", "weight 1/(|Q||S_q|)", "objective",
                   "mean token ratio"], rows))
    add("")
    add(f"- batch objective J = {_f(t['loss']['J'])}")
    add(f"- loss L = -J = {_f(t['loss']['L'])}")
    add("")
    add("Per-token log-prob ratios for every kept report:")
    add("")
    add("```")
    for entry in t["loss"]["per_report"]:
        add(f"{entry['side']}{entry['index']} new_logp="
            f"{[_f(v) for v in entry['new_logp']]}")
        add(f"{entry['side']}{entry['index']} old_logp="
            f"{[_f(v) for v in entry['old_logp']]}")
        add(f"{entry['side']}{entry['index']} ratio="
            f"{[_f(v) for v in entry['ratio']]}")
    add("```")
    add("")

    # 10. Gradient update ----------------------------------------------------
    add("## 10. Gradient update")
    add("")
    d = t["diagnostics"]
    add(f"- learning rate: {d['learning_rate']}")
    add(f"- policy epochs on this rollout batch: {d['num_policy_epochs']}")
    add(f"- gradient norm: {_f(d['grad_norm'])}")
    add("")
    params = t["parameters"]
    before = params["before"]
    after = params["after"]
    rows = [
        ["trainable parameters", before["trainable_parameters"],
         after["trainable_parameters"]],
        ["trainable elements", before["trainable_numel"],
         after["trainable_numel"]],
        ["total L2 norm", _f(before["total_l2_norm"]),
         _f(after["total_l2_norm"])],
    ]
    add(_md_table(["", "before step", "after step"], rows))
    add("")

    # 11. Before/after diagnostics -------------------------------------------
    add("## 11. Before/after diagnostic metrics")
    add("")
    add(f"- approx KL (mean old_logp − new_logp over report tokens): "
        f"{_f(d['approx_kl'])}")
    add(f"- clip fraction (tokens with |ρ − 1| > ε): {_f(d['clip_fraction'])}")
    add(
        "- mean sampled-token surprisal (entropy estimate): "
        f"{_f(d['entropy'])}"
    )
    add("")
    ckpt = t["checkpoint"]
    add(f"- checkpoint path: `{ckpt['path']}` (saved: {ckpt['saved']})")
    add("")
    env = t["environment"]
    add("Environment summary:")
    add("")
    add("```")
    add(json.dumps(env, indent=2, ensure_ascii=False))
    add("```")

    text = "\n".join(md) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text
