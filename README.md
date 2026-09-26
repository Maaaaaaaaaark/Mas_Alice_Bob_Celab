# HotpotQA 3-Agent MAS Base Experiment

A base (baseline) multi-agent experiment on HotpotQA: 100 fixed validation
questions x 10 fixed seeds = 1000 trajectories, run by three independent
logical agents (Alice, Bob, Celab) that share ONE frozen Gemma-3-1B-IT
model.

The single source of truth for the experimental design is
[hotpotqa_base_mas_experiment_spec.md](hotpotqa_base_mas_experiment_spec.md)
(v1.1). `references/` contains OPTIMA paper code that is reference-only;
where the two conflict, the spec always wins. This implementation never
reproduces OPTIMA protocols (no two-agent alternation, no `<A>` termination,
no token-pressure prompts, no reward/PPL/DPO/SFT training, no distractor
injection).

---

## 1. Directory tree and per-file purposes

```
D:\Base_Experiment_hotpot_mas\
├── hotpotqa_base_mas_experiment_spec.md  # spec v1.1 (single source of truth, unmodified)
├── README_FOR_AGENT.md                   # spec-authority notice (unmodified)
├── README.md                             # this file
├── requirements.txt                      # pinned minimum versions
├── pytest.ini                            # pytest config (testpaths=tests, pythonpath=.)
├── .gitignore                            # ignores __pycache__, .pytest_cache, outputs/
├── configs/
│   └── base.yaml                         # all experiment settings (model, generation, seeds, modes, paths)
├── prompts/                              # versioned prompt templates (sha256-hashed)
│   ├── alice_system.txt                  # Alice system prompt ($private_evidence placeholder)
│   ├── bob_system.txt                    # Bob system prompt (symmetric)
│   ├── celab_system.txt                  # Celab coordinator prompt + <TO>/<FINAL> rules
│   └── forced_final_instruction.txt      # fixed instruction used at the decision cap
├── references/                           # OPTIMA paper code (reference-only, unmodified)
├── hotpot_mas/                           # the Python package
│   ├── __init__.py                       # package marker, __version__
│   ├── config.py                         # ExperimentConfig dataclass, YAML load/validate, mode + CLI overrides, to_dict()
│   ├── prompts.py                        # PromptSet: loads/hashes templates, string.Template rendering, prompt_version
│   ├── seeds.py                          # seed_all (python/numpy/torch cpu+cuda), seed_torch (per-call re-seed)
│   ├── messages.py                       # Message / Event dataclasses + to_dict
│   ├── parser.py                         # minimal <TO>/<FINAL> parser with uniqueness rules
│   ├── evaluation.py                     # official HotpotQA evaluator port (normalize/F1/EM)
│   ├── hotpotqa.py                       # HotpotQA validation loading (datasets)
│   ├── question_selection.py             # fixed-100 selection + alphabetical evidence partition + manifest
│   ├── model_engine.py                   # ModelEngine ABC + HFEngine (shared frozen model) + MockEngine (test double)
│   ├── agents.py                         # Agent (history/input building), make_agents, check_agent_isolation
│   ├── orchestrator.py                   # per-run state machine (steps, cap, forced final, parse errors)
│   ├── logging_io.py                     # environment info + JsonlWriter (resumable append) + load_runs
│   ├── runner.py                         # trajectory loop, resume, per-run progress line
│   ├── report.py                         # run-level / per-question / dataset summaries + report.md
│   └── cli.py                            # CLI: prepare-questions / run / report
├── tests/
│   ├── conftest.py                       # sample question, config, prompts, MockEngine run helper
│   ├── test_parser.py                    # routing/termination/uniqueness parsing
│   ├── test_evaluation.py                # official evaluator consistency samples
│   ├── test_selection.py                 # selection rules, determinism, partition, manifest integrity
│   ├── test_isolation.py                 # spec sec. 18.1-18.5, 18.16 (no-GPU, via MockEngine call recording)
│   ├── test_orchestrator.py              # spec sec. 18.6-18.11, 18.15 (steps/cap/forced final/gen cap/parse error)
│   └── test_logging.py                   # spec sec. 15 schema, token-sum consistency, resume
└── outputs/                              # created at runtime (gitignored)
    ├── question_manifest.json            # fixed 100 questions + partitions + rules + sha256
    └── hotpotqa_base_mas/v3/
        ├── runs.jsonl                    # 1000 run records, one JSON object per line
        ├── summary_run_level.json        # mean/std over all runs + all rates
        ├── summary_per_question.json     # per-question means/stds
        ├── summary_dataset.json          # dataset-level (over per-question means)
        └── report.md                     # human-readable report
```

## 2. Installation (on the GPU machine)

```bash
# Python >= 3.10
pip install -r requirements.txt
```

`requirements.txt`: `torch>=2.4`, `transformers>=4.50`, `datasets>=2.19`,
`pyyaml>=6.0`, `tqdm>=4.66`, `numpy>=1.26`, `pytest>=8.0`.

`google/gemma-3-1b-it` is a gated HuggingFace model; log in once:

```bash
huggingface-cli login
```

## 3. Exact commands

```bash
# Data preparation: download HotpotQA validation and write the fixed manifest
python -m hotpot_mas.cli prepare-questions --config configs/base.yaml

# Stage 2 smoke: 1 question x 1 run (seed 0)
python -m hotpot_mas.cli run --config configs/base.yaml --mode smoke

# Stage 3 small: 5 questions x 2 runs (seeds 0,1)
python -m hotpot_mas.cli run --config configs/base.yaml --mode small

# Stage 4 full: 100 questions x 10 runs (seeds 0..9)
python -m hotpot_mas.cli run --config configs/base.yaml --mode full

# Tests (no GPU needed; MockEngine)
pytest tests/

# Re-generate the report from an existing runs.jsonl
python -m hotpot_mas.cli report --runs outputs/hotpotqa_base_mas/v3/runs.jsonl
```

CLI overrides for `run`: `--questions N`, `--runs N`, `--seeds 0,1,2`.

## 4. Output locations

| Artifact | Path |
|---|---|
| Question manifest | `outputs/question_manifest.json` |
| Raw run log (1000 records) | `outputs/hotpotqa_base_mas/v3/runs.jsonl` |
| Run-level summary | `outputs/hotpotqa_base_mas/v3/summary_run_level.json` |
| Per-question summary | `outputs/hotpotqa_base_mas/v3/summary_per_question.json` |
| Dataset summary | `outputs/hotpotqa_base_mas/v3/summary_dataset.json` |
| Human-readable report | `outputs/hotpotqa_base_mas/v3/report.md` |

Each `runs.jsonl` line is a full run record per spec sec. 15: run identity,
question + private evidence, prompt version/hashes, resolved config,
environment + engine info, the complete event trace (every event carries
`visible_history_message_ids`), and run-level results (decision steps, all
token counters, query/response counts, cap flags, termination reason, error,
F1/EM).

## 5. Reported metrics

From `report.md` / the JSON summaries (spec sec. 16): F1/EM mean ± std;
generated tokens per agent and total (primary efficiency metric,
`total_generated_tokens`); input tokens (auxiliary); decision steps; Alice
and Bob query counts; message counts; Cap Rate; Generation Cap Rate;
Natural Termination Rate; Parse Error Rate; Forced Final Rate; Error Rate;
termination-reason distribution; dataset-level stats over per-question
means; a consistency check that `total_generated_tokens` equals the sum of
`model_generation` event tokens for every run.

## 6. VRAM estimate (RTX 3090 24GB)

- Weights fp16 ≈ 2.1GB (single shared model, loaded once).
- KV cache ≈ 26KB/token (1 KV head; 4 global + 22 sliding-window-512
  layers). batch=1, input <= 16k, max_new_tokens=2048 -> KV ≈ 84MB,
  eager-attention transient ≈ 2.6GB.
- **Peak ≈ 5-7GB**, comfortably within 24GB (even a full 32k input would
  peak around 9GB).
- Runs execute sequentially with append-only logging: no memory
  accumulation across the 1000 trajectories.

## 7. Implementation assumptions (design decisions)

1. **Model calls**: transformers `AutoModelForCausalLM.generate()`, batch=1,
   `torch.inference_mode()`, `use_cache=True`; one frozen model shared by
   the three logical agents; no vLLM.
2. **Evidence content**: each worker receives the full paragraph of its
   supporting document, formatted `Title: {title}\n{paragraph}`.
3. **Partition rule**: the two distinct supporting titles are sorted
   alphabetically; first -> Alice, second -> Bob. Deterministic, no RNG,
   both supporting docs are never on one side; distractors never included.
4. **Question selection**: candidates must have a non-empty string answer,
   `type` in {bridge, comparison}, exactly 2 distinct supporting titles,
   both titles in context, both paragraphs non-empty; then
   `random.Random(sample_selection_seed=0)` shuffle, take 100. All
   rejection statistics are recorded in the manifest. "Single-side
   answerability" is not machine-verifiable (limitation, sec. 9).
5. **Messages**: the model generates the `Alice:` / `Bob:` / `Celab:`
   prefixes itself (required by the system prompts); prefixes are generated
   tokens and counted. Raw message text (including routing markers) is
   delivered verbatim to the receiver's history; worker outputs are never
   parsed, so marker-looking text inside worker replies has no effect.
6. **Parser uniqueness**: a Celab output must contain exactly one action
   type (`<TO>ALICE</TO>`, `<TO>BOB</TO>`, `<FINAL>...</FINAL>`) and that
   marker exactly once. A repeated same-type marker is ambiguous ->
   parse error. `<FINAL></FINAL>` is a valid final with an empty prediction.
   Case-sensitive; incomplete markers do not match.
7. **Decision steps**: every Celab call in the normal interaction phase
   counts +1 (including the final call and any parse-error call); worker
   replies do not count; the forced-final call does not count (steps stay
   at the cap).
8. **Forced final**: after 20 steps without a final, the fixed
   `forced_final_instruction.txt` is appended to Celab's history as a new
   user turn (controller event, generated_tokens=0) and Celab is called
   once more (`forced_final=True`, decision_step=20, tokens counted).
   If it follows a worker reply, both user-visible texts remain separate in
   structured history but are coalesced into one tokenizer turn so Gemma's
   strict user/assistant alternation remains valid.
   Parseable -> `termination_reason="forced_final"`; otherwise
   `"forced_final_parse_failure"` with `final_answer=null`.
9. **Token accounting**: `input_tokens = len(apply_chat_template(tokenize=True,
   add_generation_prompt=True))` with the real Gemma tokenizer;
   `generated_tokens = len(output_ids[prompt_len:])`, including any EOS.
   Primary metric `total_generated_tokens = alice + bob + celab` (includes
   parse-error outputs and the forced-final call). `finish_reason`:
   last generated id in the resolved EOS set (tokenizer + generation_config,
   e.g. [1, 106]) -> "eos", else "length";
   `generation_cap_reached = (no EOS and len >= max_new_tokens)`.
10. **Chat template**: the Gemma-3 template accepts `system` only as
    messages[0] (folded into the first user turn) and requires strict
    user/assistant alternation starting with user. The question is
    therefore delivered as Celab's first user turn via a controller
    `initial_task` message (spec sec. 5.3 lists Q as a separate input
    component).
11. **Seeds**: run_index k -> run_seed k for every question; `seed_all` at
    run start. Every model call receives a stable seed derived from
    `(run_seed, speaker, per-speaker call index)`, giving the three logical
    agents independent, reproducible sampling streams. Same GPU/driver/library
    versions are required for reproduction (cross-version bitwise equality is
    not guaranteed).
12. **Logging**: JSONL, one complete record per run, appended and fsynced
    immediately after each run. Run IDs include the question ID and seed;
    resume skips only records whose full trajectory fingerprint matches.
    An invalid trailing fragment is removed before appending. `num_messages`
    = model_generation event
    count; `num_alice_responses` / `num_bob_responses` recorded separately.
13. **Parse-error evaluation**: `final_answer=null` is passed to the
    official evaluator as an empty prediction (F1/EM = 0) and is never
    guessed, rewritten, retried, or repaired.
14. **Engine selection**: the orchestrator only knows the `ModelEngine`
    interface; `HFEngine` (GPU) and `MockEngine` (tests) share the same
    `generate(messages, seed, speaker)` contract.

## 8. OPTIMA code: reused vs reimplemented

| Component | Decision | Notes |
|---|---|---|
| Agent abstraction (name/memory/independent step) | Reimplemented | 3 agents + strict context isolation; no vLLM client |
| vLLM call / name-prefix prefill | Not used | transformers `generate()`; prefixes are model-generated tokens |
| `cal_f1_score` (LLM-tokenizer F1) | Not used | not the official evaluator; official normalize/F1/EM ported verbatim |
| HotpotQA field handling | Reimplemented | supporting docs only, no distractors, split across workers |
| `<A>` parser / math parser / boxed | Not used | minimal `<TO>`/`<FINAL>` parser |
| JSONL append / resumable logging | Reimplemented (idea kept) | spec sec. 15 schema |
| reward/PPL/DPO/SFT, token-pressure prompts, two-agent alternation | Not used | excluded by spec |

## 9. Known limitations

- **Single-side answerability** of the 2-document split cannot be verified
  semantically by a machine; only the supporting-facts/type/paragraph
  filters are enforced, plus the recorded rules for manual spot-checks.
- **Bit-level reproducibility boundary**: CUDA kernel implementations can
  differ across driver/library versions; identical results are expected
  only for the same GPU + driver + library versions (seeds, dtype, and
  attention implementation are all recorded per run).
- **GPU integration boundary**: CPU/mock tests and real HotpotQA loading are
  covered locally; actual Gemma generation still requires the gated model and
  a CUDA machine.
- **Strict parser**: the uniqueness rule (exactly one marker of one type)
  is deliberately strict and may inflate the Parse Error Rate; parse errors
  are recorded, never repaired (design choice, spec sec. 7).
- **Gemma-3-1B-IT is gated**: an HF token with access to the model is
  required. Model and tokenizer are pinned to a Hub commit in `base.yaml`;
  their resolved revisions are also logged at run time.
- **Dataset download**: `prepare-questions` downloads the Parquet-backed
  HotpotQA dataset from the Hugging Face Hub; no remote dataset code is run.
