#!/bin/bash
#SBATCH --job-name=smoke-setup
#SBATCH --account=aip-boyuwang
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:15:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#
# Short job: same modules + venv path as slurm_ma_tom.sh, then smoke_setup.py --gpu.
# Use this to validate the cluster stack before queuing a multi-hour train job.
#
#   cd $SCRATCH/mini-agent && mkdir -p logs
#   sbatch slurm_smoke.sh
#   sbatch --account=YOURGROUP slurm_smoke.sh
#
# Optional: also test torch.compile (needs Triton):
#   sbatch slurm_smoke.sh --compile-smoke
#
# Login node only (no GPU queue): source .venv/bin/activate && python smoke_setup.py

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$SCRIPT_DIR/hpc_job_env.sh"

export WANDB_MODE="${WANDB_MODE:-offline}"

echo "=== smoke job ==="
echo "PROJECT_ROOT=$PROJECT_ROOT  host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Python: $(which python)  Torch: $(python -c 'import torch;print(torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi failed)"

PYTHONUNBUFFERED=1 python -u smoke_setup.py --gpu "$@"
