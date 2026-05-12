#!/bin/bash -l
# ---------------------------------------------------------------------------
# One-time conda environment setup for the MDE viewpoint benchmark.
# Run this on a login node BEFORE submitting the SLURM job.
#
# Usage:
#   bash setup_env.sh
# ---------------------------------------------------------------------------
set -euo pipefail

module load conda/latest
module load cuda/12.8

ENV_PREFIX="/project/pi_dagarwal_umass_edu/project_19/daksh3/conda_envs/mde_depth310"

if [ ! -d "$ENV_PREFIX" ]; then
    echo "[setup] creating new env at $ENV_PREFIX"
    conda create -y -p "$ENV_PREFIX" python=3.10
fi

conda activate "$ENV_PREFIX"

# Use scratch as the pip cache so /home doesn't fill up.
PIP_CACHE_DIR="/scratch3/workspace/dakshsanjayk_umass_edu-wenlong_vwn/pip_cache"
mkdir -p "$PIP_CACHE_DIR"
export PIP_CACHE_DIR

# Torch + CUDA 12.1 wheels (compatible with the cuda/12.8 module).
pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Project requirements
pip install -r "$(dirname "$0")/mde_viewpoint/requirements.txt"

echo "[setup] done. Verify with:"
echo "  python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
