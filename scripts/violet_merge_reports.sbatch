#!/bin/bash
#SBATCH --job-name=hotpot-merge
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/common/home/users/y/yh.liang.2026/slurm_logs/%x.%j.out

set -euo pipefail

MAS_HOME=/common/home/users/y/yh.liang.2026
REPO_DIR="$MAS_HOME/projects/Mas_Alice_Bob_Celab"
ENV_DIR="$MAS_HOME/conda_envs/hotpot_mas"
NUM_SHARDS=4

module purge
module load Anaconda3/2024.06-1
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
[[ "$CONDA_PREFIX" == "$ENV_DIR" ]]
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"
srun python -m hotpot_mas.cli merge-shards \
  --config configs/one_shot_direct_answer_distractor.yaml \
  --mode full \
  --num-shards "$NUM_SHARDS"
srun python -m hotpot_mas.cli merge-shards \
  --config configs/single_agent_distractor.yaml \
  --mode full \
  --num-shards "$NUM_SHARDS"
