# Architecture

## Environment (`env.py`)

**ThreeCorridorVec** (primary, V3) — batch of N parallel 9×7 grid envs, pure NumPy.

```
######### row 0
#L..C..R# row 1  goals: LEFT=(1,1)  CENTER=(1,4)  RIGHT=(1,7)
##.#.#.## row 2  bottlenecks: (2,2)=L  (2,4)=C  (2,6)=R
#.......# row 3  shared corridor
#.......# row 4  shared corridor / spawn
#.......# row 5  spawn zone
######### row 6
```

- Two agents, hidden private corridor intent sampled from {LEFT=0, CENTER=1, RIGHT=2}
- 3-channel spatial obs (walls, self, partner) + 3-dim one-hot goal vector
- `mask_partner_obs`: optional flag to zero partner channel (for ablation experiments)
- No cell sharing: same-target and swap conflicts block both agents
- Bottleneck collision: both entering same bottleneck → -5 each + episode end
- `corridor_committed`: tracks first bottleneck row-2 entry per agent (-1 = uncommitted)
- Episode info includes: `partner_intent`, `chosen_corridor_a/b`, `collisions`, `successes`, `corridor_match`
- `scripted_actions_vec()`: deterministic rule-based partner (beelines to bottleneck then goal)

**ForkedCorridorVec** (legacy, V1/V2) — original 7×7 two-corridor env, kept for backward compat.

## V1 Model (`train_v1.py`)

Stateless CNN ActorCritic, parameter-shared.

```
obs (3,7,7) → Conv2d(3→32) → ReLU → Conv2d(32→64) → ReLU → flatten
→ cat(goal_vec) → Linear(3138→256) → ReLU
→ pi: Linear(256→5)   val: Linear(256→1)
```

## V2/V3 Model (`models_drc.py`)

DRC(3,3) recurrent actor-critic, parameter-shared. Supports arbitrary grid (H,W) and goal_dim.

```
obs (3,H,W) → encoder Conv2d(3→G) → ReLU → i_t

DRC Core (N ticks × D layers):
  Each layer d at tick n:
    input x = cat(i_t, skip)
      - d=0: skip = h_top from previous tick (top-down)
      - d>0: skip = h_{d-1} from current tick (inter-layer)
    pool-and-inject: mean/max pool h → linear → reshape → add to h
    h_d, c_d = ConvLSTM(x, h_d + pool_inject, c_d)

Output: cat(h_top, i_t) → flatten → cat(goal) → Linear(2*G*H*W+goal_dim→256) → ReLU
→ pi: Linear(256→5)   val: Linear(256→1)
```

Default config: D=3, N=3, G=32, H=7, W=9 (three-corridor grid).

Hidden states (h, c) shape: (D, B, G, H, W). Persist across timesteps, zeroed at episode boundaries.

Three forward paths:
- `forward()` — full forward with sampling, returns action + log_prob + entropy + value + new states
- `get_value()` — value-only pass
- `forward_logits()` — logits + value + new states (no sampling); takes optional `inject_cell_top`
- `inject_cell_top`: (B, G, H, W) tensor added to c[D-1] before core (for causal interventions)

## Training (`train.py`, V3)

CleanRL MAPPO for ThreeCorridorVec. Key params:
- GAE(λ=0.95, γ=0.97), PPO clip=0.2, LR 4e-4→0 linear decay
- Entropy 0.01, weight decay 1e-5 on heads only (per paper E.4)
- Sequence-level mini-batching with full BPTT over `num_steps`
- Hidden states zeroed where `done=True` after each env step
- Logs corridor-choice frequencies per update

Eval-only mode (`--eval-only --checkpoint X --traces-out Y`): run greedy rollouts, save traces
(cell states at probe steps + corridor metadata) for interpretability.py.

Legacy: `train_v1.py` (stateless CNN), `train_v2.py` (DRC on old 7×7 env).

## Data flow

```
env.reset() → (obs, goal_vec)
              ↓
  Model.forward(obs, goal, h, c) → action, value, new_h, new_c
              ↓
env.step(actions) → (obs, goal_vec, rewards, dones, infos) → wandb
              ↓ (zero h,c where done)
  next iteration
```

## V3 Interpretability (`interpretability.py`)

Scientific question: does A's DRC cell state at an early step linearly encode B's intended corridor,
and does injecting the concept vector causally shift A's corridor choice toward the specific injected corridor?

Pipeline:
1. `collect_rollouts()` — run greedy episodes (num_envs=1), record `c[D-1]` (layer-3 cell) for
   agent A at each of probe steps [0..5]. Primary step must be reached; later steps best-effort.

2. **Probe** (`run_probe`): 3-class logistic regression on c^D → partner corridor (L/C/R).
   Reports macro F1, per-class F1, confusion matrix (all logged to wandb).

3. **Ablation** (`run_ablation`): re-collect with `mask_partner_obs=True`, re-probe.
   Expected: F1 drops if partner channel carries the signal.

4. **Negative control** (`run_neg_control`): shuffle partner-intent labels, retrain probe.
   Expected: F1 → ~chance (0.33).

5. **Corridor-specific interventions** (`run_interventions`):
   For each target corridor k:
   - W_k = unit-normed coef_[k] × inject_scale, reshaped to (G, H, W)
   - Find rollouts where B's intent ≠ k (counterfactual condition)
   - Fork each rollout at probe_step: (a) no inject, (b) inject W_k, (c) inject random vector
   - Run to end; record A's committed corridor
   - Report P(A→k | inject W_k) vs P(A→k | base) vs P(A→k | random)
   - Key test: targeted injection should raise P(A→k) above both base and random control

6. **Emergence** (`run_emergence`): probe at steps 0–5, log macro F1 per step.
   Expected: near-chance at step 0, rising as partner evidence accumulates.

Concept vector extraction: `clf.coef_[k]` (OvR logistic regression) → W_k direction increases
classifier score for class k. Injection adds W_k to c[D-1] before ConvLSTM core runs.
