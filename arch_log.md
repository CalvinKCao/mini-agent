# Architecture Log

## 2025-04-02 — Initial V1

- `env.py`: ForkedCorridorVec, 7x7 grid, 2 agents, hidden goals, bottleneck collisions
- `train_v1.py`: CNN ActorCritic (Conv→Conv→MLP), param-shared MAPPO, CleanRL-style
- Vectorized NumPy env (512 parallel), torch.compile + bfloat16 on GPU
- wandb logging: episode returns, success/collision rates, PPO losses

## 2025-04-02 — V2: DRC integration + env hardening

- `env.py`: added no-cell-sharing rule (same-target + swap blocking between active agents); reached-goal agents removed from physics and partner obs
- `models_drc.py`: DRC(3,3) ConvLSTM actor-critic — 3 layers, 3 ticks, 32 channels, pool-and-inject, bottom-up + top-down skip connections
- `train_v2.py`: recurrent MAPPO — hidden state management across timesteps, sequence-level mini-batching with full BPTT, mask-based done resets for autograd compatibility
- `slurm_ma_tom.sh`: Slurm job script for L40S on Killarney (Alliance Canada), auto-creates venv under $PROJECT/$USER
- Weight decay applied only to policy/value heads (per paper E.4)

## 2025-04-02 — V3 interpretability

- `interpretability.py`: default ckpt `checkpoints/v2_final.pt`; record flattened top-layer **cell** state; logistic regression probe (macro F1); causal test via `inject_cell_top` on `c[D-1]` for agent A
- `models_drc.py`: `forward_logits`; optional `inject_cell_top` on `forward` / `get_value` / `forward_logits` (adds to `c[D-1]` before core)
- `requirements.txt`: `scikit-learn`
- `compile_safe.py`: skip `torch.compile` when Triton missing / compile fails (avoids `TritonMissing` on Alliance GPU nodes); Slurm venv bootstrap installs `triton` when absent
- `hpc_job_env.sh`: shared Slurm bootstrap (resolve `PROJECT_ROOT`, modules, `.venv`); sourced by `slurm_ma_tom.sh` and `slurm_smoke.sh`
- `smoke_setup.py` + `slurm_smoke.sh`: quick import/env/GPU/DRC checks before long queues

## 2026-04-06 — V3 upgrade: three-corridor environment + high-rigour interpretability

### Environment
- `env.py`: added `ThreeCorridorVec` — 9×7 grid, three bottlenecks (L/C/R), goals at (1,1)/(1,4)/(1,7); `corridor_committed` per agent; richer `infos` dict (partner_intent, chosen_corridor_a/b, corridor_match); `mask_partner_obs` flag for ablation; `scripted_actions_vec()` deterministic partner
- `ForkedCorridorVec` kept as-is for V1/V2 backward compat

### Model
- `models_drc.py`: no architecture change; H/W already parameterized; used with H=7, W=9, goal_dim=3

### Training
- `train.py` (new): DRC MAPPO for ThreeCorridorVec; 9×7 obs, 3-class goal; corridor-choice frequency logging; `--eval-only --traces-out` mode for pre-computing rollout traces
- `compile_safe.py` (new standalone): torch.compile wrapper, factored out of train scripts

### Interpretability
- `interpretability.py` (full rewrite): 3-class probe (L/C/R) on c^D; confusion matrix; partner-visibility ablation (masked partner channel); shuffled-label negative control; corridor-specific causal interventions (inject W_k, measure corridor choice Δ vs base and vs random-vector control); time-of-emergence analysis (probe at steps 0–5); held-out generalization split
