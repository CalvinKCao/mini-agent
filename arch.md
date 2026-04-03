# Architecture

## Environment (`env.py`)

`ForkedCorridorVec` — batch of N parallel 7x7 grid envs, pure NumPy.

- Two agents, hidden private goals (L or R)
- 3-channel spatial obs (walls, self, partner) + 2-dim one-hot goal vector
- No cell sharing: same-target and swap conflicts block both agents
- Bottleneck collision: both targeting same bottleneck → -5 each + episode end
- Reached-goal agents removed from grid (non-blocking, invisible to partner)
- Auto-resets done envs; tracks episode stats in `infos`

## V1 Model (`train_v1.py`)

Stateless CNN ActorCritic with parameter sharing.

```
obs (3,7,7) → Conv2d(3→32) → ReLU → Conv2d(32→64) → ReLU → flatten
→ cat(goal_vec) → Linear(3138→256) → ReLU
→ pi: Linear(256→5)   val: Linear(256→1)
```

## V2 Model (`models_drc.py` + `train_v2.py`)

DRC(3,3) recurrent actor-critic with parameter sharing.

```
obs (3,7,7) → encoder Conv2d(3→32) → ReLU → i_t

DRC Core (3 ticks × 3 layers):
  Each layer d at tick n:
    input x = cat(i_t, skip)
      - d=0: skip = h_top from previous tick (top-down)
      - d>0: skip = h_{d-1} from current tick (inter-layer)
    pool-and-inject: mean/max pool h → linear → reshape → add to h
    h_d, c_d = ConvLSTM(x, h_d + pool_inject, c_d)

Output: cat(h_top, i_t) → flatten → cat(goal) → Linear(3138→256) → ReLU
→ pi: Linear(256→5)   val: Linear(256→1)
```

Hidden states (h, c) persist across timesteps, reset on episode boundaries.
PPO update: sequence-level mini-batching with full BPTT.

## Training

Both V1 and V2: CleanRL MAPPO, GAE(λ=0.95, γ=0.97), PPO clip=0.2,
LR 4e-4→0 linear decay, entropy 0.01, weight decay 1e-5 on heads.

## Data flow

```
env.reset() → (obs, goal_vec)
              ↓
  Model.forward(obs, goal, h, c) → action, value, new_h, new_c
              ↓
env.step(actions) → (obs, goal_vec, rewards, dones, infos) → wandb
              ↓ (reset h,c where done)
  next iteration
```
