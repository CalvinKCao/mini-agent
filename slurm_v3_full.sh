#!/bin/bash
#SBATCH --job-name=v3-tom-full
#SBATCH --account=aip-boyuwang
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Full V3 pipeline on a single L40S:
#   1. Train DRC(3,3) MAPPO on three-corridor env   (~5-6h)
#   2. Run all interpretability probes               (~30-45min)
#      probe / ablation / neg-control / corridor interventions / emergence
#
# Wall-time budget (L40S):
#   5M steps, 512 envs     → ~5-6h
#   interp (1000 rollouts, 150-trial injections, GPU) → ~30min
#   Total                  → ~7h → 8h requested
#
# Usage:
#   mkdir -p logs && sbatch slurm_v3_full.sh
#   sbatch --account=OTHER-GROUP slurm_v3_full.sh
#   # Override total timesteps:
#   TOTAL_STEPS=10000000 sbatch slurm_v3_full.sh

set -euo pipefail

# ── env: modules, venv, PROJECT_ROOT, cd into repo ────────────────────
_env_sh="${SLURM_SUBMIT_DIR:-}/hpc_job_env.sh"
if [ ! -f "$_env_sh" ]; then
    echo "ERROR: hpc_job_env.sh not found at $_env_sh"
    echo "  Submit from the repo root:  cd \$SCRATCH/overcooked && sbatch slurm_v3_full.sh"
    exit 1
fi
# shellcheck source=/dev/null
source "$_env_sh"

mkdir -p logs traces
export WANDB_DIR="${PROJECT_ROOT}/wandb"
mkdir -p "$WANDB_DIR"

TOTAL_STEPS="${TOTAL_STEPS:-5000000}"
RUN_TAG="j${SLURM_JOB_ID}"

# ── diagnostics ───────────────────────────────────────────────────────
echo "==================================================================="
echo "  PROJECT_ROOT : $PROJECT_ROOT"
echo "  SLURM job    : $SLURM_JOB_ID  node=$(hostname)"
echo "  GPU          : $CUDA_VISIBLE_DEVICES"
echo "  Python       : $(which python)"
echo "  TOTAL_STEPS  : $TOTAL_STEPS"
echo "==================================================================="
python -c '
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
name = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  device: {name}")
'
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ── phase 1: training ─────────────────────────────────────────────────
echo ""
echo "=== PHASE 1: TRAINING (${TOTAL_STEPS} steps) ==="
date

PYTHONUNBUFFERED=1 python -u train.py \
    --num-envs       512 \
    --total-timesteps "$TOTAL_STEPS" \
    --device         cuda \
    --run-name       "v3-drc33-${RUN_TAG}"

echo "Training done at $(date)"
ls -lh checkpoints/v3_final.pt

# ── phase 2: interpretability ─────────────────────────────────────────
echo ""
echo "=== PHASE 2: INTERPRETABILITY ==="
date

# Run all modes in one pass; model is loaded once, rollouts collected fresh.
# 1000 rollouts × 6 probe steps + ablation (400) + 150-trial interventions.
PYTHONUNBUFFERED=1 python -u interpretability.py \
    --checkpoint          checkpoints/v3_final.pt \
    --num-rollouts        1000 \
    --n-intervention-trials 150 \
    --probe-step          2 \
    --inject-scale        1.0 \
    --device              cuda \
    --run-name            "v3-interp-${RUN_TAG}"

echo "Interpretability done at $(date)"

echo ""
echo "=== JOB COMPLETE ==="
date
