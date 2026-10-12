#!/bin/bash
# Stage 1a: shared worker SFT (docs/WORKER_SFT.md).
#
# Prepares the worker data (unless it already exists) and runs the SFT
# training in the requested mode on one GPU. Submit from the repository
# root so SLURM_SUBMIT_DIR resolves correctly.
#
#   sbatch scripts/violet_worker_sft.sbatch                  # trace (default)
#   WORKER_SFT_MODE=smoke sbatch scripts/violet_worker_sft.sbatch
#   WORKER_SFT_MODE=normal sbatch scripts/violet_worker_sft.sbatch
#
# GPU / time / resource overrides (do not edit mid-experiment without
# recording a new experiment version):
#   sbatch --constraint=l40s scripts/violet_worker_sft.sbatch
#   sbatch --time=12:00:00 --mem=32G scripts/violet_worker_sft.sbatch

#SBATCH --job-name=worker-sft
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/common/home/users/y/yh.liang.2026/slurm_logs/%x.%j.out

set -euo pipefail

MAS_HOME=/common/home/users/y/yh.liang.2026
MAS_SCRATCH=/common/scratch/users/y/yh.liang.2026
REPO_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
ENV_DIR="$MAS_HOME/conda_envs/hotpot_mas"
CONFIG=configs/worker_sft.yaml
MODE="${WORKER_SFT_MODE:-trace}"

trap 'echo "worker-sft job finished with exit code $?"' EXIT

module purge
module load Anaconda3/2024.06-1
eval "$(conda shell.bash hook)"
conda activate "$ENV_DIR"
[[ "$CONDA_PREFIX" == "$ENV_DIR" ]]

export HF_HOME="$MAS_SCRATCH/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"

echo "==== environment ===="
which python
python --version
python - <<'PY'
import torch, transformers, peft, datasets
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("datasets", datasets.__version__)
PY

echo "==== GPU ===="
srun whichgpu
srun nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "==== worker-sft data prep (mode: $MODE) ===="
PREP_ARGS=(--config "$CONFIG" --mode "$MODE")
if [[ "${WORKER_SFT_FORCE_DATA:-0}" == "1" ]]; then
  PREP_ARGS+=(--force)
fi
srun python -m hotpot_mas.training.cli prepare-worker-sft-data "${PREP_ARGS[@]}"

echo "==== worker-sft training (mode: $MODE) ===="
RESUME_ARGS=()
if [[ "${WORKER_SFT_RESUME:-0}" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi
srun python -m hotpot_mas.training.cli train-worker-sft \
  --config "$CONFIG" \
  --mode "$MODE" \
  "${RESUME_ARGS[@]}"

echo "==== outputs ===="
echo "Exact paths are printed by the CLI above (final_checkpoint /"
echo "worked_example / stage_dir). In trace mode the worked example is:"
echo "  outputs/worker_sft/worker_sft/trace/worked_example_worker_sft.md"
echo "  outputs/worker_sft/worker_sft/trace/step_*/adapter/"
echo "  outputs/worker_sft_data/trace/ (prepared data + dedup + label reports)"
