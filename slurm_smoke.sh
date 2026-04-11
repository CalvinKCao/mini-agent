#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --account=aip-boyuwang
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# 20-minute sanity check: build node-local venv, verify imports, GPU, env step,
# model forward.  Should take ~5-8 min total (venv build ~3-5 min, tests <1 min).
#
# Usage:
#   mkdir -p logs && sbatch slurm_smoke.sh

set -euo pipefail

_env_sh="${SLURM_SUBMIT_DIR:-}/hpc_job_env.sh"
if [ ! -f "$_env_sh" ]; then
    echo "ERROR: hpc_job_env.sh not found — submit from the repo root"
    exit 1
fi
source "$_env_sh"

echo "=== SMOKE TEST ==="
echo "Job $SLURM_JOB_ID  node=$(hostname)  gpu=$CUDA_VISIBLE_DEVICES"
echo "venv: $VIRTUAL_ENV"
date

python - <<'EOF'
import sys, torch, numpy as np
from sklearn.linear_model import LogisticRegression

# ── GPU ──────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "CUDA not available"
print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  device: {torch.cuda.get_device_name(0)}")

# ── env ───────────────────────────────────────────────────────────────
from env import ThreeCorridorVec, GRID_H, GRID_W, N_CORRIDORS, N_ACTIONS
env = ThreeCorridorVec(num_envs=4, max_steps=10, seed=0)
obs, gv = env.reset()
assert obs.shape == (4, 2, 3, GRID_H, GRID_W), f"bad obs shape {obs.shape}"
obs, gv, rew, done, info = env.step(env.rng.integers(0, N_ACTIONS, size=(4, 2)))
print(f"env OK  obs={obs.shape}")

# ── model ─────────────────────────────────────────────────────────────
from models_drc import DRCActorCritic
dev = torch.device("cuda")
m = DRCActorCritic(obs_ch=3, goal_dim=N_CORRIDORS, G=32, D=3, N=3,
                   H=GRID_H, W=GRID_W).to(dev)
h, c = m.initial_state(4, device=dev)
o = torch.zeros(4, 3, GRID_H, GRID_W, device=dev)
g = torch.zeros(4, N_CORRIDORS, device=dev)
logits, val, nh, nc = m.forward_logits(o, g, h, c)
assert logits.shape == (4, 5)
print(f"model forward OK  logits={logits.shape}  val={val.shape}")

# ── sklearn ───────────────────────────────────────────────────────────
clf = LogisticRegression(max_iter=50).fit(
    np.random.randn(30, 10), np.random.randint(0, 3, 30)
)
print(f"sklearn OK  classes={clf.classes_}")

# ── checkpoint load ───────────────────────────────────────────────────
import os
ckpt = "checkpoints/v3_final.pt"
if os.path.isfile(ckpt):
    state = torch.load(ckpt, map_location=dev, weights_only=False)
    print(f"checkpoint OK  ({ckpt})")
else:
    print(f"checkpoint not found ({ckpt}) — skipping (run train.py first)")

print("\n=== ALL CHECKS PASSED ===")
EOF

date
echo "Smoke test complete."
