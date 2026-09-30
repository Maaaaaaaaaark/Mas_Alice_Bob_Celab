#!/bin/bash
#SBATCH --job-name=hotpot-single-full
#SBATCH --partition=pradeepresearch
#SBATCH --account=pradeepresearch
#SBATCH --qos=pradeepresearch-priority
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000
#SBATCH --time=4-23:00:00
#SBATCH --array=0-3%4
#SBATCH --requeue
#SBATCH --output=/common/home/users/y/yh.liang.2026/slurm_logs/%x.%A_%a.out

set -euo pipefail

MAS_HOME=/common/home/users/y/yh.liang.2026
MAS_SCRATCH=/common/scratch/users/y/yh.liang.2026
REPO_DIR="$MAS_HOME/projects/Mas_Alice_Bob_Celab"
ENV_DIR="$MAS_HOME/conda_envs/hotpot_mas"
NUM_SHARDS=4

module purge
module load Anaconda3/2024.06-1
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
[[ "$CONDA_PREFIX" == "$ENV_DIR" ]]

export HF_HOME="$MAS_SCRATCH/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"
srun whichgpu
srun nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -m hotpot_mas.cli run \
  --config configs/single_agent_distractor.yaml \
  --mode full \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SLURM_ARRAY_TASK_ID"
