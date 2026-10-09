#!/bin/bash
set -euo pipefail

MAS_HOME=/common/home/users/y/yh.liang.2026
MAS_SCRATCH=/common/scratch/users/y/yh.liang.2026
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="$MAS_HOME/conda_envs/hotpot_mas"

module purge
module load Anaconda3/2024.06-1
eval "$(conda shell.bash hook)"

mkdir -p "$MAS_HOME/conda_envs" "$MAS_HOME/slurm_logs"
mkdir -p "$MAS_SCRATCH/hf_cache"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  conda create \
    --prefix "$ENV_DIR" \
    --override-channels \
    --channel conda-forge \
    python=3.11 pip -y
fi

conda activate "$ENV_DIR"
python -m pip install --upgrade pip
python -m pip install \
  torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install \
  transformers==4.57.6 \
  'datasets>=2.19,<5' \
  'pyyaml>=6.0,<7' \
  'tqdm>=4.66,<5' \
  'numpy>=1.26,<3' \
  'peft>=0.12,<1' \
  'pytest>=8.0,<10'

cd "$REPO_DIR"
python -m pip check
pytest -q

echo "Environment ready: $CONDA_PREFIX"
python -c 'import sys, torch, transformers; print(sys.executable); print("torch", torch.__version__, "cuda", torch.version.cuda); print("transformers", transformers.__version__)'
