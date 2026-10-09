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
| Worker nucleus sampling | not specified | worker rollout requires `top_p=1.0`; temperature sampling remains enabled | a changing top-p support can assign zero probability to an old report token after an update, producing `-inf` log probabilities and infinite KL diagnostics |

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
| `fake_policy.py` | offline test doubles: `FakePolicy` (categorical θ with real autograd), `FakeTokenizer`, `FakeChatTokenizer` (offset-carrying chat-template double for SFT tests), `FakeSynthesizer` / `FakeTrainableSynthesizer` (scripted / frozen-base+LoRA C that run the real parse+F1 code) |
| `cross_pair.py` | pure functions: R → Q_A/Q_B, population std, signal flags, normalized advantages |
| `grpo_loss.py` | pure tensor functions: ratio, clip, per-report objective, batch J/L, approx KL, clip fraction, entropy |
| `rollout.py` | one question end to end: G×2 sampled reports, G×G C calls (i-major), signal filtering, empty-report exclusion |
| `eval.py` | deterministic worker evaluation (F1/EM) + per-checkpoint report cache (Stage 2 reuses identical reports, incl. empty-report control) |
| `worker_trainer.py` | Stage 1 loop: collect N signal questions → padded variable-length minibatches → periodic val → restore the best checkpoint by val F1 for final test → `metrics.jsonl` |
| `sft_data.py` | Stage 2 data: `build_sft_example` — render the chat template once, re-encode with char offsets, locate the gold-answer token span inside the wrapper, verify prefix stability, mask everything but gold tokens (`-100`) |
| `sft_loss.py` | Stage 2 loss: causal next-token shift, per-example `1/\|y*\|` averaging (TeX Eq. 4), NaN-safe padding mask, non-finite guards |
| `synthesizer_trainer.py` | Stage 2 loop: resolve the validation-best worker checkpoint → freeze workers → deterministic cached A/B reports → SFT C (independent LoRA) on gold tokens only → per-epoch val → restore best C → 4-condition final comparison (see §7) |
| `trace.py` | `trace.json` recording/validation + `worked_example.md` renderer (reads trace.json only, deterministic) |
| `cli.py` | `prepare-data`, `train-workers --mode trace|smoke|pilot|full [--resume]`, `train-synthesizer --mode trace|smoke`, `render-trace` |

Configuration lives in `configs/cross_paired_grpo_workers.yaml` (Stage 1)
and `configs/synthesizer_sft.yaml` (Stage 2); all G / N / seed / lr /
paths are config-controlled (nothing hardcoded in code). Prompts live in
`prompts_training/`.

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

# Stage 2 trace (1/1/1 questions, 1 epoch; frozen workers resolve from the
# Stage 1 SMOKE run — Stage 1 trace records no validation F1, see §7)
python -m hotpot_mas.training.cli train-synthesizer \
  --config configs/synthesizer_sft.yaml --mode trace

# Stage 2 smoke (64/32/32 questions, 2 epochs; frozen workers resolve from
# the Stage 1 smoke run's validation-best checkpoint)
python -m hotpot_mas.training.cli train-synthesizer \
  --config configs/synthesizer_sft.yaml --mode smoke
```

`pilot` / `full` use the same commands with `--mode pilot|full`. **Do not
launch `full` during development** — it is the production-scale run
(2000 train questions, G=4, T=200).

Outputs land under `outputs/training/<experiment_id>/<experiment_version>/`
(checkpoints `step_XXXX/`, `metrics.jsonl`, `eval_cache/`, and in trace
mode `trace.json` + `worked_example.md`).

## 7. Stage 2 (synthesizer SFT) — implemented

TeX Algorithm 2: freeze the Stage 1 workers and SFT-train synthesizer C.

**Pipeline** (all in `hotpot_mas/training/synthesizer_trainer.py`):

1. **Resolve the worker checkpoint** — `worker_checkpoint` may be a Stage 1
   run dir or a `step_XXXXX/adapter` dir. A run dir resolves the *best* step
   by `mean_val_f1` from `metrics.jsonl` (strict `>`, earliest max — the same
   update rule as `worker_trainer`) and cross-checks the final `state.json`
   (`best_step`/`best_val_f1` must match exactly); a mismatch or a missing
   best adapter is a hard error — Stage 2 never falls back to untrained
   workers. If the adapter carries `adapter_config.json`, its base model /
   LoRA r / alpha are checked against the Stage 2 config.
2. **Freeze the workers** — every worker parameter `requires_grad_(False)`;
   any remaining trainable parameter is a hard error.
3. **Deterministic cached A/B reports** — one greedy-decode report per
   (question, side) with the *worker-evaluation* decode settings, cached per
   split under `report_cache/<identity>.json`. The identity is a hash over
   {worker checkpoint, split, decode params, prompt version, prompt hashes},
   so a different checkpoint/config never reuses stale reports. C0 (step 0)
   and Cφ (final test) always read the *same* cache; the training cache is
   separate from the val/test caches.
4. **SFT on gold-answer tokens only** — C shares the Gemma-3-1B base with
   its own fresh LoRA φ (r=16, α=32, dropout=0, all-linear; a fresh adapter
   equals the base model, so C0 is the step-0 evaluation — enforced at load
   time by a zero-LoRA check). `build_sft_example` renders the chat template
   once, re-encodes with char offsets, locates the gold-answer token span
   *inside* the wrapper, and masks everything else — question, A/B report,
   `Celab: <FINAL>` and closing `</FINAL>` (prefix stability of the
   tokenizer is verified; if no token lies fully inside the answer span the
   run fails instead of supervising the whole completion). The loss uses the
   causal next-token shift and per-example `1/|y*|` averaging (Eq. 4).
5. **Loop** — per minibatch: `zero_grad → backward → step`, only C's LoRA
   gets gradients; per epoch: validation on the fixed cached reports
   (F1/EM/C tokens, plus the empty-report control), save `epoch_NNN/`, save
   `best_checkpoint/` on improvement. After the last epoch the best C is
   *restored* (never the last epoch). Memory: the worker model is deleted
   before C is loaded (single A5000).
6. **Final comparison** — on the identical cached test reports:
   C0 + reports, Cφ + reports, C0 + empty, Cφ + empty → mean F1 / EM / C
   generated tokens / parse counts, plus A/B token counts (from the cache,
   not counted as C's tokens), in `final_comparison.json` + `report.md`.

**Outputs** land under
`outputs/training/synthesizer_sft/<mode>/`:
`config_snapshot.json`, `worker_checkpoint_provenance.json`,
`report_cache_metadata.json`, `report_cache/`, `step0_baseline.json`,
`metrics.jsonl` (fields: `epoch, train_loss, grad_norm, mean_val_f1,
mean_val_em, mean_val_c_tokens, empty_val_f1, empty_val_em, best_val_f1,
checkpoint_path`), `epoch_NNN/`, `best_checkpoint/`,
`final_comparison.json`, `report.md`.

**Known wrinkle**: Stage 1 *trace* runs skip validation entirely (no
validation F1 is ever recorded), so the Stage 2 trace config resolves its
frozen workers from the Stage 1 *smoke* run's best checkpoint instead.

## 8. Known limitations

- The CPU-only test suite (Stage 1 + Stage 2: best-checkpoint selection,
  frozen-worker/cache reuse, gold-only SFT masking, no causal off-by-one,
  best-C restore, empty-report control, non-finite guards, CLI parsing)
  runs offline; a real Gemma trace/smoke run still requires the GPU
  cluster and remains the verification gate for model loading, PEFT
  integration, CUDA placement, and GPU memory use. The Stage 2 suite has
  not been executed on the development machine (no Python interpreter
  there) — run it in the cluster environment.
- The Stage 2 worker-adapter compatibility check is skipped when the
  adapter carries no `adapter_config.json` (the Stage 1 adapter dirs do
  carry it, so this only affects hand-made checkpoints).
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
