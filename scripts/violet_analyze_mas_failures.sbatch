#!/bin/bash
#SBATCH --job-name=hotpot-failure-analysis
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/common/home/users/y/yh.liang.2026/slurm_logs/%x.%j.out

set -euo pipefail

MAS_HOME=/common/home/users/y/yh.liang.2026
REPO_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
ENV_DIR="$MAS_HOME/conda_envs/hotpot_mas"

RUNS_PATH="${1:?Usage: sbatch $0 RUNS_PATH OUTPUT_DIR}"
OUTPUT_DIR="${2:?Usage: sbatch $0 RUNS_PATH OUTPUT_DIR}"

module purge
module load Anaconda3/2024.06-1
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
[[ "$CONDA_PREFIX" == "$ENV_DIR" ]]
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"
srun python -m hotpot_mas.cli analyze-failures \
  --runs "$RUNS_PATH" \
  --output-dir "$OUTPUT_DIR"
