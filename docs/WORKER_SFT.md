# Stage 1a: Independent Worker SFT (the NEW experiment)

This document is the specification for the NEW Stage 1a experiment:
supervised fine-tuning (SFT) of ONE shared worker model on local
single-document questions.

**Scope note.** Older documents in this repository say things like
"workers cannot see the question", "distractors are not allowed", or
"only C is trained with SFT". Those rules belong to the OLDER
experiments (Stage 1 cross-paired GRPO workers, Stage 2 synthesizer SFT)
and still apply to them. They do NOT apply to Stage 1a. This document,
together with the code in `hotpot_mas/training/worker_sft/` and
`configs/worker_sft.yaml`, defines Stage 1a behavior. The old
experiments and their saved results are preserved and untouched.

## 1. What is being trained

- **One shared worker model.** The same model will later be used by
  workers A and B. There is no coordinator C in this stage, no RL, no
  dynamic routing, no joint training.
- **Model:** a FRESH LoRA adapter on `google/gemma-3-1b-it`
  (default `r=16`, `alpha=32`, `dropout=0.05`). The exact model id and
  revision are verified and recorded in every run. Old GRPO /
  synthesizer adapters are never loaded by default.
- **Input:** one explicit local question plus one or more visible
  documents (the single-document phase trains with 1 document; a later
  multi-document phase uses 1–5 documents, easy distractors first).
  Document order is shuffled per example and the source ids are
  remapped to display ids (`D0`, `D1`, ...) so the model cannot learn
  position or storage order.
- **Output format (exactly):**
  - One to two complete sentences copied VERBATIM from the documents,
    one line per source id: `D3: "exact sentence"`.
  - Two different ids only when the check verifies that both documents
    are required; otherwise one line.
  - Exactly `NOT FOUND` when the question cannot be answered from the
    visible documents (and "not answerable" is not the same as "needs
    more than two sentences" — such examples are excluded and the reason
    is recorded).
  - No ANSWER/SOURCE sections, no reasoning, no paraphrasing.
  - The short answer and its aliases are metadata ONLY. They are used
    for checks and evaluation and are never put into the prompt.

## 2. Training task in one sentence

Learn `(local question + visible documents) -> target`, where the target
is verbatim source sentences with display ids, or exactly `NOT FOUND`.

## 3. Data sources and conversion

| Source | Split used | How it becomes worker examples |
|---|---|---|
| SQuAD 2.0 | train | Official answerable rows: positives whose evidence is the sentence containing the official answer span (with answer offsets for verification; original text is kept untouched; a second sentence is prepended only when the evidence sentence cannot stand alone because of a pronoun/reference). Official unanswerable rows: `NOT FOUND` negatives. |
| MuSiQue | train | One example per decomposition subquestion of answerable parent rows. `#N` references are resolved to the answers of EARLIER subquestions (the original and resolved question plus every replacement are recorded; the current subquestion's answer is never inserted). Sibling paragraphs that do not contain the answer become negatives with the same resolved question. |
| HotpotQA | train only | No automatic candidates. Examples come only from the reviewed file (or the optional teacher-generated local-question file). Supporting facts define the evidence; teacher outputs are checked and kept separate, and are loadable from a reviewed local-question file. There is no live teacher call per run; a missing teacher file is reported clearly, never invented and never silently omitted. |

- Source weights are configurable; the FINAL mixture after filtering and
  dedup is reported per run.
- Mixture target: about 70% answerable / 30% unanswerable after
  filtering and dedup. A negative means "unanswerable from ALL visible
  documents" (absence of the answer string is not proof of that — the
  checks and review exist because of this). Not all negatives come from
  SQuAD. Quotas are met by selection, never by duplicating samples.
- The 30% target is a TRAINING mixture target; evaluation splits are not
  forced to it.

## 4. Label quality and leakage checking

- **Automatic checks** (run on every example): the target sentence is
  verbatim in its stated document, ids are valid, at most two sentences,
  evidence contains the answer or an alias, character offsets are
  consistent, no unresolved `#N` references, nothing would be truncated,
  and the rendered target matches the output format. Failures exclude
  the example with a recorded reason.
- **Review status is separate.** Auto-checked is NEVER reported as
  human-verified. The review status records the reviewer, method, and
  provenance. Weak labels never silently enter training
  (`require_reviewed_for_training` is on in trace mode).
- **Split isolation:** val/test definitions are frozen BEFORE training
  variants are built, at the PARENT level (all variants of one original
  example land in the same split). The official MuSiQue exclusion file
  (`data/dev_test_singlehop_questions_v1.0.json`, sha256-verified) is
  applied first. Checks cover original ids, parent ids, source-dataset
  ids, normalized question/subquestion matches, normalized paragraph
  matches, near-duplicates (documented Jaccard method), distractor
  paragraphs too, and overlap with the project's existing eval
  manifests. A training candidate that conflicts with a fixed eval
  sample is removed. Eval paragraphs never appear in training.
- **Dedup report:** every run writes counts, methods, thresholds, and
  rejection reasons. String matching is reported honestly as
  "reduces risk", never as "proves zero leakage".

## 5. How the SFT loss works (plain English + the equation)

**Teacher forcing.** The model reads the full prompt. At every position
inside the gold assistant turn it is asked "what is the next token?".
The gold text is fed in directly; nothing is generated and then
aligned.

For one example with supervised positions `P`:

```
L_example = (1/|P|) * sum over p in P of  -log P(correct token at p | prefix)
```

That is: the average, over the supervised target tokens, of the
negative log-probability of the correct token given everything before
it. The update loss for one optimizer step is the mean of the example
losses in its accumulation group.

**Causal alignment:** a causal LM's logits at position `t` predict the
token at position `t+1`, so the code aligns `logits[:, :-1]` with
`labels[:, 1:]`. The first target token is predicted from the last
prompt position, and the end-of-turn token is predicted from the last
content token; the code asserts BOTH are trained.

**Masking (`-100` = ignored):** system/user prompt tokens, document
tokens inside the prompt, padding, and any non-target template tokens.
The ENTIRE assistant turn — first target token through the end-of-turn
token — is supervised, because the worker must learn the format, the
document ids, the exact quotation text, and where the turn ends. The
Stage 2 synthesizer SFT (mask everything except a gold answer inside
Celab/`<FINAL>` wrappers) is deliberately NOT copied here.

**Parameter updates:** only LoRA parameters are passed to the optimizer
(AdamW, defaults `lr=1e-5`, `weight_decay=0.0`, `betas=[0.9, 0.999]`,
`eps=1e-8`, `scheduler=none`, `warmup=0.0`, clip `1.0`, `bf16` when
supported, seed 0, 1 epoch). Every effective value is recorded in the
run's config snapshot. The base model is verified frozen twice: a
`requires_grad` audit at load time and a sha256 hash of the base
parameters before and after training.

**Gradient accumulation:** microbatch = 1 example; each microbatch
backward is scaled by `1/group_size`, where the group sizes are planned
BEFORE training from the deterministic example order (normal run
`grad_accum=8`; trace `1`). The final partial group therefore keeps
every example's exact weight — the loss never depends on where the
group boundary falls.

## 6. Run modes

| Mode | Purpose |
|---|---|
| `trace` | The first real run: 10 examples (7 answerable / 3 unanswerable, all three sources when valid prepared samples exist, at least one two-sentence target, single-document by default), microbatch 1, grad-accum 1, 1 epoch = exactly 10 optimizer steps. Requires reviewed training examples. Writes the full structured trace and `worked_example_worker_sft.md` (sections A–J). Missing requirements are reported honestly — never fabricated. |
| `smoke` | Larger functional check (default 64 examples) with grad-accum. |
| `normal` | The real training run (default 4096 examples, grad-accum 8). |

The trace also runs a small fixed before/after generation comparison
(same inputs, greedy decoding). The 10 steps are NOT expected to
improve outputs, and the worked example says so.

## 7. Commands

Run from the repository root with the `hotpot_mas` conda environment:

```
# one-time per VIOLET_CLUSTER.md (never put the token in a script or the repo):
hf auth login

# data prep (trace mode: 10 reviewed examples)
python -m hotpot_mas.training.cli prepare-worker-sft-data \
  --config configs/worker_sft.yaml --mode trace

# tokenizer-only length report (no model weights, no training)
python -m hotpot_mas.training.cli inspect-worker-sft-lengths \
  --config configs/worker_sft.yaml --mode trace

# train (first run: trace = 10 optimizer steps)
python -m hotpot_mas.training.cli train-worker-sft \
  --config configs/worker_sft.yaml --mode trace

# re-render the worked example from the trace, any time, no GPU:
python -m hotpot_mas.training.cli render-worker-sft-trace \
  --trace outputs/worker_sft/worker_sft/trace/trace.json
```

Cluster (from the repository root; the script checks the GPU it
actually received and prints exact output paths):

```
sbatch scripts/violet_worker_sft.sbatch                  # trace (default)
WORKER_SFT_MODE=smoke sbatch scripts/violet_worker_sft.sbatch
WORKER_SFT_MODE=normal sbatch scripts/violet_worker_sft.sbatch
WORKER_SFT_RESUME=1 sbatch scripts/violet_worker_sft.sbatch   # continue
```

## 8. Outputs (new directories, separate from GRPO / C-SFT)

- `outputs/worker_sft_data/<version>/` — prepared data
  (`prepared_train/val/test.jsonl`, `probes.jsonl`, `candidates.jsonl`),
  `dedup_report.json`, `label_quality_report.json`, `data_manifest.json`,
  `splits.json`, `lengths_report.json` (after inspect).
- `outputs/worker_sft/worker_sft/<version>/` — `config_snapshot.json`,
  `metrics.jsonl`, `fixed_example_outputs.json`, per-step checkpoints
  (`step_*/adapter/` + `state.json`; resume continues from the latest),
  and in trace mode `trace.json` +
  `worked_example_worker_sft.md`.

Every run gets a unique run id; existing artifacts are never silently
overwritten (`--force`/`--resume` are explicit).

## 9. The worked example (`worked_example_worker_sft.md`)

Rendered ONLY from the structured `trace.json` (deterministic — the
same trace always renders the same Markdown; values are never invented
by hand). Sections:

- A. Run identity (date, run id, code revision, model/tokenizer
  revision, hardware, library versions, effective config and seeds,
  real-model vs fixture)
- B. Raw data (dataset/split/original id/parent id, full original
  question and visible text, answer annotations clearly marked
  "not shown to the model")
- C. Data conversion (original vs resolved local question,
  substitutions, offsets, 1-vs-2-sentence target, why negative,
  auto-checks and review status)
- D. Dedup and sample selection (checks, counts, removed samples with
  reasons, exact selected ids)
- E. Actual worker inputs and targets (exact role messages, rendered
  prompt, all documents actually seen, exact assistant target)
- F. Tokenization and the loss mask (counts; token-level table:
  position, token id, readable token, label, contributes-to-loss;
  prompt masking + end-of-turn supervision; causal alignment)
- G. SFT loss (teacher forcing in plain English; per supervised target
  token log-prob and NLL for representative examples; the arithmetic to
  the example loss and how example losses form the update loss; values
  from the ACTUAL training forward)
- H. Parameter updates (microsteps vs optimizer steps, lr, losses, grad
  norm before/after clip, trainable tensor/element counts,
  frozen/trainable, LoRA parameter-change norm, how
  base-weights-unchanged is checked, checkpoint paths)
- I. Before/after generation (same fixed inputs and decoding, reference
  target, actual pre and post outputs, parsing/source/quote-match/
  answer-coverage/abstention checks, token counts, finish status;
  failures included)
- J. Limits and next steps (what is confirmed, what is not established,
  what is deferred)

## 10. Not claimed, not part of this stage

- The trace run does not establish a quality improvement, a baseline,
  or evaluation numbers. Full baseline/large evaluations are a separate
  later step.
- A fake-model test (the CPU-only test suite) is a functional test only;
  it is recorded as `real_model=False` and is not a training result.
- The multi-document training phase (1–5 docs) is deferred; only fixed
  multi-document probes exist today.
