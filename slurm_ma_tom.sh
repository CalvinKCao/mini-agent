#!/bin/bash
#SBATCH --job-name=forked-corridor-tom
#SBATCH --account=aip-boyuwang
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=0-06:00:00
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
# For H100 (large runs only):
#   sbatch --partition=gpubase_h100_b4 --gpus-per-node=h100:1 \
#          --gres=NONE --mem=64G slurm_ma_tom.sh

set -euo pipefail

# ── resolve project root ─────────────────────────────────────────────
# Prefer the directory you submitted from (sbatch from repo root → works).
# Otherwise try common clone locations (GitHub repo name is mini-agent).
_resolve_root() {
    local d
    for d in \
        "${SLURM_SUBMIT_DIR:-}" \
        "${SCRATCH:-}/mini-agent" \
        "${SCRATCH:-}/overcooked" \
        "${HOME}/mini-agent" \
        "${HOME}/overcooked"
    do
        [ -z "$d" ] && continue
        d=$(readlink -f "$d" 2>/dev/null) || continue
        if [ -f "$d/train_v2.py" ]; then
            echo "$d"
            return 0
        fi
    done
    return 1
}

if ! PROJECT_ROOT=$(_resolve_root); then
    echo "ERROR: could not find repo root (need train_v2.py)."
    echo "  Clone to e.g. \$SCRATCH/mini-agent, then:"
    echo "    cd \$SCRATCH/mini-agent && mkdir -p logs && sbatch slurm_ma_tom.sh"
    exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"

# ── modules ──────────────────────────────────────────────────────────
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load cudnn/8.9

# ── venv (under $PROJECT/$USER to keep $SCRATCH clean) ───────────────
if [ -z "${PROJECT:-}" ] && [ -d "$HOME/projects" ]; then
    FIRST_PROJ=$(ls -d "$HOME"/projects/def-* "$HOME"/projects/aip-* 2>/dev/null | head -1)
    [ -n "$FIRST_PROJ" ] && export PROJECT=$(readlink -f "$FIRST_PROJ")
fi

# venv next to the checkout (same layout as local .venv)
VENV_DIR="${PROJECT_ROOT}/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR ..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip -q
    pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
    pip install numpy wandb -q
else
    source "$VENV_DIR/bin/activate"
fi

cd "$PROJECT_ROOT"

# ── run ──────────────────────────────────────────────────────────────
echo "Job $SLURM_JOB_ID  node=$(hostname)  gpu=$CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)  Torch: $(python -c 'import torch;print(torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHONUNBUFFERED=1 python -u train_v2.py \
    --num-envs 512 \
    --total-timesteps 5000000 \
    --device cuda \
    "$@"
