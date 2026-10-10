#!/bin/bash
#SBATCH --job-name=mas-learner-checks
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
REPO_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
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

REWARD_KIND="${REWARD_KIND:?Export the selected REWARD_KIND before submission}"
case "$REWARD_KIND" in
  f1|em|gold_mean_log_likelihood) ;;
  *) echo "Unsupported REWARD_KIND=$REWARD_KIND" >&2; exit 2 ;;
esac
MODE="overfit_${REWARD_KIND}"
echo "Learner checks use reward_kind=$REWARD_KIND (mode=$MODE)"

srun python -m hotpot_mas.training.cli prepare-data \
  --config configs/learner_checks.yaml \
  --mode "$MODE"

# Gate 1: set -e stops the job here on any mismatch.
srun python -m hotpot_mas.training.cli check-logprobs \
  --config configs/learner_checks.yaml \
  --mode "$MODE" \
  --mean-tolerance 2e-3 \
  --max-tolerance 2e-2

# Gate 2: the command fails after writing overfit_result.json unless the
# fixed 32-question training reward rises above its step-0 value.
srun python -m hotpot_mas.training.cli train-workers \
  --config configs/learner_checks.yaml \
  --mode "$MODE"
