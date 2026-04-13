#!/bin/bash
#SBATCH --job-name=ma-tom-interp
#SBATCH --account=aip-boyuwang
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Run interpretability.py on a trained checkpoint.
# Fast on GPU (~5 min compute); venv build adds ~2-3 min.
# 20-min wall is generous — typical L40S queue is very short at this size.
#
# Usage (from $SCRATCH/mini-agent):
#   mkdir -p logs && sbatch slurm_interp.sh
#
# Override defaults at submit time:
#   sbatch slurm_interp.sh --probe-step 4 --inject-scale 2.0
#
# To run a specific mode only (faster):
#   sbatch slurm_interp.sh --mode intervention --probe-step 4

set -euo pipefail

_env_sh="${SLURM_SUBMIT_DIR:-}/hpc_job_env.sh"
if [ ! -f "$_env_sh" ]; then
    echo "ERROR: hpc_job_env.sh not found at $_env_sh"
    echo "  Submit from the repo root: cd \$SCRATCH/mini-agent && sbatch slurm_interp.sh"
    exit 1
fi
source "$_env_sh"

export WANDB_DIR="${PROJECT_ROOT}/wandb"
mkdir -p "$WANDB_DIR"

echo "Job $SLURM_JOB_ID  node=$(hostname)  gpu=$CUDA_VISIBLE_DEVICES"
echo "torch $(python -c 'import torch;print(torch.__version__)')  device: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

CKPT="${PROJECT_ROOT}/checkpoints/v3_final.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    echo "  Train first with: sbatch slurm_ma_tom.sh  (or point --checkpoint at v3_step*.pt)"
    exit 1
fi

PYTHONUNBUFFERED=1 python -u interpretability.py \
    --checkpoint "$CKPT" \
    --num-rollouts 1000 \
    --probe-step 4 \
    --intervention-step 4 \
    --inject-scale 1.0 \
    --n-intervention-trials 200 \
    --device cuda \
    "$@"
