#The Markdown experiment specification is the source of truth.
 
The OPTIMA paper/code files are provided only as references.
 
Do not reproduce OPTIMA's experimental protocol blindly.
 
In particular:

- this experiment has THREE independent agents: Alice, Bob, Celab;

- Celab is the coordinator;

- Alice/Bob never receive the original question directly;

- do not include HotpotQA distractors;

- use <TO>ALICE</TO>, <TO>BOB</TO>, and <FINAL>...</FINAL>;

- do not use OPTIMA's <A>...</A> termination rule;

- do not implement OPTIMA training/reward/PPL logic;

- do not add communication-efficiency prompts;

- follow the Markdown specification whenever it conflicts with reference code.

---

SCOPE NOTE (added 2026-10-12, not part of the original notice above):
the rules above apply to the ORIGINAL inference-baseline experiment only.
The NEW Stage 1a experiment (one shared worker model, SFT, local question
plus visible documents, verbatim source-sentence answers or NOT FOUND) is
specified in docs/WORKER_SFT.md, which overrides any old rule that
conflicts with it. The original experiment and its saved results are
unchanged.
 