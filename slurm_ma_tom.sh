#!/bin/bash
#SBATCH --job-name=forked-corridor-tom
#SBATCH --account=aip-boyuwang
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# DRC MAPPO training on Forked Corridor (Theory-of-Mind PoC).
# Default: L40S on Killarney.
#
# Usage (from repo root on scratch, e.g. $SCRATCH/mini-agent):
#   mkdir -p logs && sbatch slurm_ma_tom.sh
#   sbatch slurm_ma_tom.sh --total-timesteps 10000000
#   sbatch --account=OTHER-GROUP slurm_ma_tom.sh
#
# Before a long run, validate setup with a short GPU job:
#   sbatch slurm_smoke.sh
#
# For H100 (large runs only):
#   sbatch --partition=gpubase_h100_b4 --gpus-per-node=h100:1 \
#          --gres=NONE --mem=64G slurm_ma_tom.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/hpc_job_env.sh"

# W&B files land here (not under logs/). Override if you want e.g. $SLURM_TMPDIR.
export WANDB_DIR="${PROJECT_ROOT}/wandb"
mkdir -p "$WANDB_DIR"

# ── run ──────────────────────────────────────────────────────────────
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "WANDB_DIR=$WANDB_DIR"
echo "PWD=$(pwd)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "Job $SLURM_JOB_ID  node=$(hostname)  gpu=$CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)  Torch: $(python -c 'import torch;print(torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHONUNBUFFERED=1 python -u train_v2.py \
    --num-envs 512 \
    --total-timesteps 5000000 \
    --device cuda \
    "$@"
