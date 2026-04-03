# Forked Corridor ToM — Onboarding

PoC: mechanistic evidence of emergent Theory of Mind in model-free RL (inspired by Bush et al. 2025). Two agents in a 7x7 grid coordinate through bottlenecks to reach private hidden goals.

## File tree

```
├── env.py              # Vectorized ForkedCorridor env (NumPy), no cell sharing
├── models_drc.py       # DRC(3,3) ConvLSTM actor-critic (pool-and-inject, skips)
├── train_v1.py         # V1: CNN ActorCritic + MAPPO baseline
├── train_v2.py         # V2: DRC MAPPO with sequence-level BPTT
├── slurm_ma_tom.sh     # Slurm job script for L40S on Alliance Canada
├── requirements.txt    # torch, numpy, wandb
├── onboard.md          # this file
├── arch.md             # architecture snapshot
├── arch_log.md         # append-only arch changelog
├── context.md          # original project spec (V1/V2/V3 phases)
├── bush_paper.txt      # reference paper (Bush et al. 2025)
└── checkpoints/        # saved model weights
```

## Planned (V3)

- `interpretability.py` — linear probes on DRC cell states to decode partner goal, causal intervention ("inception test")

## Key design choices

- Grid: wide shared corridor (rows 3-5), bottlenecks at (2,2)/(2,4), goals at (1,1)/(1,5).
- No cell sharing: agents block each other everywhere; bottleneck collision = -5 + episode end.
- Reached-goal agents are removed from physics (no blocking, hidden from partner obs).
- Parameter-shared MAPPO for both V1 (stateless CNN) and V2 (recurrent DRC).
- V2 PPO update: sequence-level mini-batching with full BPTT over rollout length.
- wandb project: `forked-corridor-tom`.

## Cluster

- Default GPU: L40S on Killarney (`slurm_ma_tom.sh`)
- Clone to `$SCRATCH/overcooked`, venv under `$PROJECT/$USER/overcooked/venv`
- `sbatch slurm_ma_tom.sh` (or override with `"$@"` args)
