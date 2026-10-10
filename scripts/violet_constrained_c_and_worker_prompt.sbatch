#!/bin/bash
#SBATCH --job-name=mas-c-worker-fix
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
OLD_CACHE="$REPO_DIR/outputs/diagnostics/inference_diagnostic/val300/untrained_worker_reports.json"
OLD_SUMMARY="$REPO_DIR/outputs/diagnostics/inference_diagnostic/val300/summary.json"

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
[[ -f "$OLD_CACHE" ]] || {
  echo "Missing cached worker reports: $OLD_CACHE" >&2
  exit 2
}
[[ -f "$OLD_SUMMARY" ]] || {
  echo "Missing old diagnostic summary: $OLD_SUMMARY" >&2
  exit 2
}

# C-only rerun: four conditions, exact same old worker reports.
srun python -m hotpot_mas.training.cli diagnose-inference \
  --config configs/inference_diagnostic_300_constrained.yaml \
  --report-cache "$OLD_CACHE" \
  --baseline-summary "$OLD_SUMMARY"

# Revised worker prompt: generate a fresh cache, then evaluate the same table.
srun python -m hotpot_mas.training.cli diagnose-inference \
  --config configs/inference_diagnostic_300_constrained.yaml \
  --mode revised_workers \
  --baseline-summary "$OLD_SUMMARY"
