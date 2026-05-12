#!/bin/bash -l
#SBATCH --job-name=mde_a100_wenlong
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --account=pi_wenlongzhao_umass_edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH -o /project/pi_dagarwal_umass_edu/project_19/daksh3/Depth-Estimation/logs/slurm-%j.out

# ---------------------------------------------------------------------------
# MDE Viewpoint Benchmark — A100 / Wenlong account
# Identical pipeline to the H100 script; only the GPU constraint differs.
# ---------------------------------------------------------------------------

set -euo pipefail

module load conda/latest
module load cuda/12.8
conda activate /project/pi_dagarwal_umass_edu/project_19/daksh3/conda_envs/mde_depth310

PROJECT_DIR="/project/pi_dagarwal_umass_edu/project_19/daksh3/Depth-Estimation"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"

export MDE_BASE_DIR="/scratch3/workspace/dakshsanjayk_umass_edu-wenlong_vwn/mde_viewpoint"
mkdir -p "$MDE_BASE_DIR"/{datasets,results,checkpoints,hf_cache,torch_cache}

export HF_HOME="$MDE_BASE_DIR/hf_cache"
export TRANSFORMERS_CACHE="$MDE_BASE_DIR/hf_cache/transformers"
export HUGGINGFACE_HUB_CACHE="$MDE_BASE_DIR/hf_cache/hub"
export TORCH_HOME="$MDE_BASE_DIR/torch_cache"

# Wire datasets, results, checkpoints to scratch.
mkdir -p "$PROJECT_DIR/datasets"
ln -sfn "$MDE_BASE_DIR/datasets/hypersim"     "$PROJECT_DIR/datasets/hypersim"
ln -sfn "$MDE_BASE_DIR/datasets/nyu_depth_v2" "$PROJECT_DIR/datasets/nyu_depth_v2"
ln -sfn "$MDE_BASE_DIR/datasets/kitti"        "$PROJECT_DIR/datasets/kitti"
ln -sfn "$MDE_BASE_DIR/datasets/tartanair"    "$PROJECT_DIR/datasets/tartanair"

[ -L "$PROJECT_DIR/results" ]     || { rm -rf "$PROJECT_DIR/results";     ln -sfn "$MDE_BASE_DIR/results"     "$PROJECT_DIR/results"; }
[ -L "$PROJECT_DIR/checkpoints" ] || { rm -rf "$PROJECT_DIR/checkpoints"; ln -sfn "$MDE_BASE_DIR/checkpoints" "$PROJECT_DIR/checkpoints"; }

echo "=========================================="
echo "Step 1 — build viewpoint-bucketed splits"
echo "=========================================="
OMP_NUM_THREADS=1 python -m mde_viewpoint.data.dataset_builder \
    --config mde_viewpoint/configs/config.yaml

echo "=========================================="
echo "Step 2 — zero-shot benchmark (4 models x 5 categories)"
echo "=========================================="
OMP_NUM_THREADS=1 python -m mde_viewpoint.experiments.run_zero_shot_eval \
    --config mde_viewpoint/configs/config.yaml

echo "=========================================="
echo "Step 3 — failure-mode analysis (DAV2 on top-down category D)"
echo "=========================================="
OMP_NUM_THREADS=1 python -m mde_viewpoint.experiments.run_failure_analysis \
    --config mde_viewpoint/configs/config.yaml \
    --model dav2 --category D --n_images 50

echo "=========================================="
echo "Step 4 — LoRA adaptation curves on category D (rank 8)"
echo "=========================================="
OMP_NUM_THREADS=1 python -m mde_viewpoint.experiments.run_lora_adaptation \
    --config mde_viewpoint/configs/config.yaml \
    --category D --rank 8

echo "=========================================="
echo "Step 5 — forgetting eval on NYU + KITTI"
echo "=========================================="
ADAPTER_DIR="$PROJECT_DIR/results/lora/category_D_rank8/N_49/best_adapter"
if [ -d "$ADAPTER_DIR" ]; then
    OMP_NUM_THREADS=1 python -m mde_viewpoint.experiments.run_forgetting_eval \
        --config mde_viewpoint/configs/config.yaml \
        --adapter_path "$ADAPTER_DIR" \
        --rank 8
else
    echo "[skip] $ADAPTER_DIR not found — run step 4 first."
fi

echo "=========================================="
echo "Job complete. Results under: $MDE_BASE_DIR/results"
echo "=========================================="
