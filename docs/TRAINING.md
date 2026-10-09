# Cross-Paired GRPO Training Experiments (Stage 1 Workers)

This document covers the **training** experiments added on top of the
inference baseline. The algorithm is specified in
[`cross_paired_grpo.tex`](../cross_paired_grpo.tex) (Algorithm 1 = worker
training, Algorithm 2 = synthesizer SFT).

## 1. Document precedence

For everything in this experiment, the precedence is:

1. **`cross_paired_grpo.tex`** — the sole algorithm specification.
2. **The user requirements recorded for this pipeline** (this document).
3. The old baseline documents (`README_FOR_AGENT.md`,
   `hotpotqa_base_mas_experiment_spec.md`) — **inference baseline only**.
   They describe an experiment that never trains, never uses the distractor
   partition, and never feeds the original question to the workers. None of
   that applies to the training runs; they are left unmodified and their
   pipeline is unchanged.

`cross_paired_grpo.pdf` is a layout rendering of the TeX only (it could not
be diffed locally: no poppler on the development machine). Where TeX, the
PDF, or the requirements conflict, the conflict is pointed out instead of
silently picking a side — see the next section.

## 2. Documented deviations / extensions of the TeX

These were agreed with the requirements rather than chosen silently:

| Point | TeX Algorithm 1 | This implementation | Why |
|---|---|---|---|
| Policy epochs per rollout batch | one update | `num_policy_epochs` configurable, **default 1 (= TeX-exact)** | with 1 epoch the ratio is always 1.0 and clipping can never be observed; epochs > 1 (trace mode uses 2) let the clip fraction be measured |
| No-signal guard | not specified | `max_sampling_attempts` bounds the question re-sampling per update (default 100); a starved update is skipped and recorded | prevents an endless loop when no question signals |
| Zero-length reports | not specified | still scored by C on all G×G pairs (they are legitimate evaluation inputs), but excluded from S_q / the token loss and recorded under `excluded_empty_reports` | a zero-token report has no tokens to carry a gradient |

Everything else follows the TeX exactly: G×G frozen-C calls per question,
row/column marginal rewards, per-side signal test `std(Q_side) > δ`
(population std, divide by G), kept set S_q = signal sides only, advantage
normalization `(Q − mean(Q)) / (std(Q) + ε_n)` within each signalling side,
token-level clipped objective on **report tokens only**
(`ρ_t = exp(new − old)`, `ℓ_t = min(ρ_t·u, clip(ρ_t, 1−ε, 1+ε)·u)`,
`J = (1/|Q|) Σ_q (1/|S_q|) Σ_reports (1/|o|) Σ_t ℓ_t`, `L = −J`), **no KL
penalty** in the loss (approx KL `mean(old − new)` is recorded as a
diagnostic only), and a frozen C that decodes greedily as the deterministic
reward evaluator (official HotpotQA token F1 as reward v1).

## 3. Data

- **Partition**: oracle-balanced using the gold supporting-document
  metadata shipped with HotpotQA. This is **not** an official HotpotQA
  agent split; the fact is recorded in `splits.json`
  (`partition_note`) and in the trace. Each worker receives 1 supporting
  document + 4 distractors (8 distractors split 4/4); per-side order is
  shuffled deterministically by `partition_seed`. Supporting labels exist
  only in manifest metadata / `document_pool` for audit and **never** enter
  any model prompt.
- **Split isolation**: train is selected first; val excludes every train
  question id; test excludes every train and val id (recorded under
  `excluded_split_overlap`). The val/test boundary is fully determined by
  the per-split selection seeds (`val_test_split_seed` is provenance).
  Disjointness is asserted before `splits.json` is written.
- **Official test split**: has no public gold answers and is not used as
  the local test set. Defaults: official `train` for training; the answered
  official `validation` split deterministically divided into val/test with
  configured counts and seeds (printed before every run).
- The gold answer `y*` enters **only** the reward/evaluation code. It is
  never part of an A/B/C prompt or input.

## 4. Module responsibilities (`hotpot_mas/training/`)

| Module | Responsibility |
|---|---|
| `config.py` | `TrainingConfig` / `SynthesizerTrainingConfig` dataclasses, YAML loading, `modes:` deep-merge, strict validation, `stage_dir()` |
| `data.py` | three fixed manifests + `splits.json`, data-configuration fingerprint, split isolation asserts, pre-run split report |
| `prompts_builder.py` | chat messages for A/B (role prompt + private evidence) and C (question + A report + B report); gold never included |
| `policy.py` | `Policy` protocol; `HFPolicy` (base model + one shared peft LoRA, manual KV-cache sampling loop, causally aligned teacher forcing, and matching rollout/re-score distributions); optimizer-state remap for resume |
| `fake_policy.py` | offline test doubles: `FakePolicy` (categorical θ with real autograd), `FakeTokenizer`, `FakeSynthesizer` (scripted C that runs the real parse+F1 code) |
| `cross_pair.py` | pure functions: R → Q_A/Q_B, population std, signal flags, normalized advantages |
| `grpo_loss.py` | pure tensor functions: ratio, clip, per-report objective, batch J/L, approx KL, clip fraction, entropy |
| `rollout.py` | one question end to end: G×2 sampled reports, G×G C calls (i-major), signal filtering, empty-report exclusion |
| `eval.py` | deterministic worker evaluation (F1/EM) + per-checkpoint report cache (Stage 2 reuses identical reports, incl. empty-report control) |
| `worker_trainer.py` | Stage 1 loop: collect N signal questions → padded variable-length minibatches → periodic val → restore the best checkpoint by val F1 for final test → `metrics.jsonl` |
| `synthesizer_trainer.py` | **Stage 2, not yet implemented** (see §7) |
| `trace.py` | `trace.json` recording/validation + `worked_example.md` renderer (reads trace.json only, deterministic) |
| `cli.py` | `prepare-data`, `train-workers --mode trace|smoke|pilot|full [--resume]`, `render-trace` |

Configuration lives in `configs/cross_paired_grpo_workers.yaml` (all G / N /
seed / lr / paths config-controlled; nothing hardcoded in code). Prompts
live in `prompts_training/`.

## 5. Trace mode

`train-workers --mode trace` runs one training question, G=2, N=1, one
rollout batch, `num_policy_epochs=2` (so clipping is observable), fixed
seed. Outputs in the stage directory:

- `trace.json` — the exhaustive per-field trace (question/gold, 10-document
  pool with gold labels, partition, actual prompts, tokenized lengths,
  decode params, G reports per side with token ids/text/old-policy
  logprobs, all G×G C inputs/outputs, normalized pred/gold, precision /
  recall / F1, full R, Q_A/Q_B, both stds + signal flags, normalized
  advantages, S_q, per-report ratio/objective, batch J/L, approx KL, clip
  fraction, entropy, grad norm, LoRA parameter summary before/after,
  checkpoint + environment). No full-vocab logits, no weight copies.
- `worked_example.md` — auto-generated English walkthrough; every number
  comes from `trace.json`, never hand-fabricated. Re-render any time:
  `python -m hotpot_mas.training.cli render-trace --trace <trace.json>`.
- `metrics.jsonl` — one line for the update (val fields null, with a note
  that validation evaluation is skipped in trace mode).

## 6. Running (on the GPU cluster)

```bash
# CPU-only unit tests (no GPU, no model download)
python -m pytest tests/ -q                # old baseline tests must stay green
python -m pytest tests/test_training_*.py -q

# Stage 1 real-model trace (Gemma-3-1B-IT, ~1-2 GB LoRA trainable)
python -m hotpot_mas.training.cli prepare-data \
  --config configs/cross_paired_grpo_workers.yaml --mode trace
python -m hotpot_mas.training.cli train-workers \
  --config configs/cross_paired_grpo_workers.yaml --mode trace

# Stage 1 smoke (tiny real run: forward + reward + backward + checkpoint)
python -m hotpot_mas.training.cli prepare-data \
  --config configs/cross_paired_grpo_workers.yaml --mode smoke
python -m hotpot_mas.training.cli train-workers \
  --config configs/cross_paired_grpo_workers.yaml --mode smoke

# Resuming from the latest checkpoint
python -m hotpot_mas.training.cli train-workers \
  --config configs/cross_paired_grpo_workers.yaml --mode smoke --resume
```

`pilot` / `full` use the same commands with `--mode pilot|full`. **Do not
launch `full` during development** — it is the production-scale run
(2000 train questions, G=4, T=200).

Outputs land under `outputs/training/<experiment_id>/<experiment_version>/`
(checkpoints `step_XXXX/`, `metrics.jsonl`, `eval_cache/`, and in trace
mode `trace.json` + `worked_example.md`).

## 7. Stage 2 (synthesizer SFT) — status

Not yet implemented. Per the plan it follows after Stage 1 has been
verified end to end on the cluster: freeze the best worker checkpoint
(selected on validation F1), give C its own independent LoRA, SFT on
input = question + A report + B report with target = gold answer only
(loss on answer tokens only), select the best C by validation F1, and
compare original C0 vs trained Cφ on identical cached reports (F1 / EM /
generated tokens) plus the empty-report control. The `Synthesizer`
interface, the eval report cache, and `SynthesizerTrainingConfig` already
exist; `synthesizer_trainer.py`, `configs/synthesizer_sft.yaml` and the
`train-synthesizer` CLI subcommand are intentionally not wired up yet.

## 8. Known limitations

- The complete CPU-only test suite passes locally. A real Gemma trace/smoke
  run still requires the GPU cluster and remains the verification gate for
  model loading, PEFT integration, CUDA placement, and GPU memory use.
- `cross_paired_grpo.pdf` was not diffed word by word (no poppler locally);
  the TeX is authoritative.
- The worker partition is oracle-balanced from gold supporting metadata —
  it is not an official HotpotQA agent assignment.
- Bitwise reproducibility holds under the same GPU/library versions
  (same boundary as `hotpot_mas/seeds.py`); CUDA kernels are not guaranteed
  bit-reproducible across GPU generations or CUDA versions.
- Large-scale runs (`pilot`, `full`) have **not been run yet**; the
  learning-curve diagnostics in `metrics.jsonl` are the instrument for
  observing them once they are.
