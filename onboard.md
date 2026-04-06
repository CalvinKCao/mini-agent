# Forked Corridor ToM — Onboarding

PoC: mechanistic evidence of latent partner-intent representation in model-free MARL.
Agents coordinate through bottlenecks to reach private hidden goals; we test whether A encodes
a decodable belief about B's intended corridor and whether injecting that belief causally changes A's route.

## File tree

```
├── env.py              # ThreeCorridorVec (9×7, 3 corridors, V3 primary) +
│                       #   ForkedCorridorVec (7×7, 2 corridors, V1/V2 legacy)
│                       #   scripted_actions_vec() for debugging/controlled rollouts
├── models_drc.py       # DRC(3,3) ConvLSTM actor-critic (pool-and-inject, BU+TD skips)
│                       #   forward_logits + inject_cell_top for probe/intervention
├── compile_safe.py     # torch.compile wrapper (graceful fallback to eager)
├── train.py            # V3: MAPPO for ThreeCorridorVec, corridor-choice logging,
│                       #   --eval-only mode to save rollout traces
├── train_v1.py         # V1: stateless CNN MAPPO baseline (7×7 env)
├── train_v2.py         # V2: DRC MAPPO on 7×7 env (legacy)
├── interpretability.py # V3 analysis: probe / ablation / neg-control /
│                       #   corridor interventions / time-of-emergence
├── slurm_ma_tom.sh     # Slurm job script for L40S on Alliance Canada
├── requirements.txt    # torch, numpy, wandb, scikit-learn
├── onboard.md          # this file
├── arch.md             # architecture snapshot
├── arch_log.md         # append-only arch changelog
├── context.md          # original project spec
├── bush_paper.txt      # reference paper (Bush et al. 2025)
└── checkpoints/        # saved model weights
```

## Quick start

```bash
# train V3 (three-corridor DRC MAPPO)
python train.py --num-envs 512 --total-timesteps 5000000

# run all interpretability tests
python interpretability.py --checkpoint checkpoints/v3_final.pt

# optional: save traces for faster re-runs
python train.py --eval-only --checkpoint checkpoints/v3_final.pt \
                --traces-out traces/v3_eval.npz --num-eval-eps 1000
python interpretability.py --traces-file traces/v3_eval.npz
```

## Key design choices

- **Grid**: 9×7, three bottleneck passages at (2,2)/(2,4)/(2,6), goals L/C/R in row 1.
- **No cell sharing**: same-target/swap conflicts block both agents; bottleneck collision = -5 + end.
- **Corridor commitment**: first entry into row 2 records the chosen corridor (-1 = uncommitted).
- **Parameter-shared MAPPO**: both agents share one DRCActorCritic.
- **BPTT**: sequence-level mini-batching over full rollout length.
- **Probe**: 3-class logistic regression on flattened c^D (layer-3 cell state) at step 2.
- **Intervention hook**: `inject_cell_top` adds a tensor to c[D-1] before ConvLSTM core.
- **Concept vector**: `clf.coef_[k]` for corridor k, unit-normed, reshaped to (G, H, W).
- **wandb project**: `forked-corridor-tom`.

## Cluster (Killarney / L40S)

- Slurm job: `sbatch slurm_ma_tom.sh`
- Clone repo to `$SCRATCH/overcooked`; venv under `$PROJECT/$USER/overcooked/venv`
- Smoke test before long jobs: `source .venv/bin/activate && python env.py`

## Gotchas

- `v3_final.pt` in `checkpoints/` is an **untrained** placeholder — train properly before probing.
- Intervention Δ will be near zero until the model is trained (untrained agents don't commit corridors).
- On CPU, BPTT is ~1s/iter for B=8; use GPU for real training.
- `train_v1.py` / `train_v2.py` still use `ForkedCorridorVec` (7×7, 2-class goal); they don't work with the new model dims.
