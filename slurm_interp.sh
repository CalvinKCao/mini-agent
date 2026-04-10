#!/bin/bash
#SBATCH --job-name=v3-interp
#SBATCH --account=aip-boyuwang
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Interpretability-only job (run after training finishes).
# Uses checkpoints/v3_final.pt from a prior train run.
#
# Usage:
#   mkdir -p logs && sbatch slurm_interp.sh
#   sbatch slurm_interp.sh --run-name v3-interp-retry   # extra args → interpretability.py

set -euo pipefail

_env_sh="${SLURM_SUBMIT_DIR:-}/hpc_job_env.sh"
if [ ! -f "$_env_sh" ]; then
    echo "ERROR: hpc_job_env.sh not found — submit from the repo root"
    exit 1
fi
source "$_env_sh"

mkdir -p logs
export WANDB_DIR="${PROJECT_ROOT}/wandb"
mkdir -p "$WANDB_DIR"

echo "Job $SLURM_JOB_ID  node=$(hostname)  gpu=$CUDA_VISIBLE_DEVICES"
python -c 'import torch; print(f"torch {torch.__version__}  device: {torch.cuda.get_device_name(0)}")'

PYTHONUNBUFFERED=1 python -u interpretability.py \
    --checkpoint          checkpoints/v3_final.pt \
    --num-rollouts        1000 \
    --n-intervention-trials 150 \
    --probe-step          2 \
    --device              cuda \
    --run-name            "v3-interp-j${SLURM_JOB_ID}" \
    "$@"
