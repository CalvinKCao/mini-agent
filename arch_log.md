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
