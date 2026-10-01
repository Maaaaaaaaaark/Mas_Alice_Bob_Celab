# Running the distractor experiments on Violet

This guide is specific to the `yh.liang.2026` account and the current Violet
allocation. GPU inference is always submitted through Slurm. The login node is
used only for Git, environment setup, authentication, and job submission.

## 1. Clone or update the repository

```bash
mkdir -p /common/home/users/y/yh.liang.2026/Project
cd /common/home/users/y/yh.liang.2026/Project
git clone https://github.com/Maaaaaaaaaark/Mas_Alice_Bob_Celab.git
cd Mas_Alice_Bob_Celab
```

For an existing clone:

```bash
cd /common/home/users/y/yh.liang.2026/Project/Mas_Alice_Bob_Celab
git pull --ff-only origin main
```

## 2. Create and verify the Conda environment

The setup script loads `Anaconda3/2024.06-1`, creates a Python 3.11 environment
at `/common/home/users/y/yh.liang.2026/conda_envs/hotpot_mas`, installs the
CUDA 12.4 PyTorch build and pinned Transformers version, activates that exact
environment, and runs the complete CPU test suite.

```bash
bash scripts/violet_setup_env.sh
```

Authenticate to Hugging Face once from the same environment. Never put the
token in a Slurm script or Git repository.

```bash
module purge
module load Anaconda3/2024.06-1
eval "$(conda shell.bash hook)"
conda activate /common/home/users/y/yh.liang.2026/conda_envs/hotpot_mas
export HF_HOME=/common/scratch/users/y/yh.liang.2026/hf_cache
hf auth login
```

Every supplied job script repeats the module load and Conda activation and
asserts the exact `CONDA_PREFIX` before running Python.

## 3. Prepare the full distractor manifest

Create the log directory before the first `sbatch`; Slurm cannot create a
missing parent directory for `--output`.

```bash
mkdir -p /common/home/users/y/yh.liang.2026/slurm_logs
cd /common/home/users/y/yh.liang.2026/Project/Mas_Alice_Bob_Celab
sbatch scripts/violet_prepare_manifest.sbatch
```

The Slurm scripts use `SLURM_SUBMIT_DIR`, so submit them from the repository
root. They do not depend on the repository being stored under a particular
case-sensitive directory name.

After completion, verify the manifest:

```bash
module purge
module load Anaconda3/2024.06-1
eval "$(conda shell.bash hook)"
conda activate /common/home/users/y/yh.liang.2026/conda_envs/hotpot_mas
python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/question_manifest_distractor.json")
manifest = json.loads(path.read_text(encoding="utf-8"))
print("questions:", manifest["num_questions_selected"])
print("evidence partition:", manifest["evidence_partition"])
print("partition seed:", manifest["partition_seed"])
print("sha256:", manifest["questions_sha256"])
PY
```

Expected: 7,345 selected questions from 7,405 validation rows,
`balanced_distractor`, partition seed 0. The manifest records why the 60 rows
that cannot satisfy the fixed two-gold/eight-distractor topology were excluded.

## 4. Run both smoke tests

```bash
sbatch scripts/violet_smoke_mas.sbatch
sbatch scripts/violet_smoke_single.sbatch
```

Both jobs request one A5000. The MAS process separately loads Alice, Bob, and C
as three model objects on logical `cuda:0`; the single-reader process loads one
model. Check status and logs:

```bash
myqueue
ls -lh /common/home/users/y/yh.liang.2026/slurm_logs
tail -n 100 /common/home/users/y/yh.liang.2026/slurm_logs/hotpot-mas-smoke.*.out
tail -n 100 /common/home/users/y/yh.liang.2026/slurm_logs/hotpot-single-smoke.*.out
```

Do not submit the full arrays until both smoke jobs finish with exit code 0.
The MAS log must report three independent physical model instances, all on
`cuda:0`, and the run record must report `physical_model_instances: 3`.

## 5. Submit the full four-shard arrays

Each array has four tasks. Every task requests one A5000 and processes a
disjoint round-robin question shard. All ten seeds for one question remain in
the same shard. Shard JSONL files are append-only and resumable after
preemption.

```bash
MAS_JOB=$(sbatch --parsable scripts/violet_full_mas_array.sbatch)
SINGLE_JOB=$(sbatch --parsable scripts/violet_full_single_array.sbatch)
echo "MAS array: $MAS_JOB"
echo "Single-reader array: $SINGLE_JOB"
```

The two arrays may be submitted together. Violet will run only as many tasks as
the available A5000 resources permit.

Expected shard sizes for 7,345 questions and ten seeds are 18,370, 18,360,
18,360, and 18,360 runs. Inspect progress with:

```bash
wc -l outputs/hotpotqa_base_mas/v11_one_shot_direct_answer_distractor/shards/*/runs.jsonl
wc -l outputs/hotpotqa_single_agent_distractor/v1/shards/*/runs.jsonl
```

## 6. Merge only after both arrays succeed

The safest submission uses an `afterok` dependency. For Slurm arrays, this
waits for every task in each array to finish successfully.

```bash
MERGE_JOB=$(sbatch --parsable \
  --dependency=afterok:${MAS_JOB}:${SINGLE_JOB} \
  scripts/violet_merge_reports.sbatch)
echo "Merge job: $MERGE_JOB"
```

The merge command validates every expected run ID before writing a total file.
It refuses incomplete, duplicated, misassigned, wrong-version, or
missing-fingerprint records. The total `runs.jsonl` is replaced atomically only
after validation succeeds.

Final reports:

```text
outputs/hotpotqa_base_mas/v11_one_shot_direct_answer_distractor/report.md
outputs/hotpotqa_single_agent_distractor/v1/report.md
```

Each condition must contain exactly 73,450 merged runs. Detailed merge audits
are written to `merge_summary.json` beside each report.

## 7. Partition choice

Although `myinfo` may list `pradeepresearch`, the scheduler can still reject
the user's group on that partition. The supplied smoke scripts therefore use
`researchshort`, and the full GPU arrays use:

```bash
#SBATCH --partition=researchlong
#SBATCH --qos=research-1-qos
```

Both use `research-1-qos`. Keep `--account=pradeepresearch`, `--requeue`, and
the A5000 constraint. Do not change partition or GPU type in the middle of an
experiment without recording the change as a new experiment version.
