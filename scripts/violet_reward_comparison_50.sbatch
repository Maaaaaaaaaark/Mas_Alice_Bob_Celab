#!/bin/bash
#SBATCH --job-name=mas-reward-compare
#SBATCH --partition=researchlong
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/common/home/users/y/yh.liang.2026/slurm_logs/%x.%j.out

set -euo pipefail
MAS_HOME=/common/home/users/y/yh.liang.2026
MAS_SCRATCH=/common/scratch/users/y/yh.liang.2026
REPO_DIR="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
ENV_DIR="$MAS_HOME/conda_envs/hotpot_mas"

module purge
module load Anaconda3/2024.06-1
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
[[ "$CONDA_PREFIX" == "$ENV_DIR" ]]
export HF_HOME="$MAS_SCRATCH/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO_DIR"
srun whichgpu
srun nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -m hotpot_mas.training.cli prepare-data \
  --config configs/reward_comparison_50.yaml
srun python -m hotpot_mas.training.cli compare-rewards \
  --config configs/reward_comparison_50.yaml
