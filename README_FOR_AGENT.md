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
 