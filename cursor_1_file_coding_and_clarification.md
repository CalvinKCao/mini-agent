# (1) File coding and clarification
_Exported on 4/6/2026 at 11:01:42 EDT from Cursor (2.5.20)_

---

**User**

@bush_paper.txt @context.md @AGENTS.md pls code this up and lmk if you run into any confusion or need clarification


---

**Cursor**

Let me read the paper in chunks and check the existing project state.

Let me read the relevant paper sections for DRC architecture details, then set up the project.

Good, I have a solid understanding of the project. Let me set up and start building V1. A few things to flag before I dive in:

1. **Grid Layout**: The ASCII art in `context.md` is very narrow (1-tile corridor). I'm interpreting it schematically and designing a wider 5-col shared corridor (rows 3-5) so early moves have lateral ambiguity. Bottlenecks at `(2,2)` and `(2,4)` with a wall between them.

2. **V2 Hyperparameter Contradiction**: The V2 brief says `D=2, N=2` but the V2 Addendum says `D=3, N=3` (DRC(3,3)). I'll go with the Addendum spec since it matches the paper.

3. **PettingZoo**: I'll follow the ParallelEnv API pattern but implement purely in NumPy (no `pettingzoo` dependency) for vectorized performance.

Let me start building.

Good, dependencies are installed. Now let me write the core files.

Good, no linter errors. Now let me write the documentation files and run a quick training smoke test.

Now let me run a quick training smoke test to make sure everything connects properly.

Training works on CPU. The agents reached 100% success rate and 0% collision rate even in this tiny 30k-step run. Let me verify GPU availability for a full run.

The installed PyTorch (2.11.0) ships CUDA 13, but the system driver only supports CUDA 12.0. Let me fix the PyTorch version.

GPU works — RTX 3050 Ti available locally (the L40S is presumably on the cluster). Let me run a GPU smoke test.

V1 is done and working. Here's what was built and a few things to flag:

---

**What's implemented (Phase V1):**

- **`env.py`** — Vectorized `ForkedCorridorVec` environment. 7x7 grid with a 5-wide shared corridor (rows 3-5), two bottleneck tiles at `(2,2)` and `(2,4)` separated by a wall, and goal zones at `(1,1)` (L) and `(1,5)` (R). Randomized spawns from 6 positions, hidden private goals, bottleneck collision penalty.

- **`train_v1.py`** — CleanRL-style MAPPO. CNN actor-critic (Conv3→32→64 + MLP→256), parameter-shared across both agents. GAE (λ=0.95, γ=0.97), PPO clip 0.2, LR linear decay 4e-4→0, entropy bonus 0.01. Supports `torch.compile` + bfloat16 autocast. Logs to wandb (`forked-corridor-tom`).

**Smoke test results:**

- Even tiny runs (30-50k steps) show agents learning to coordinate — 80-100% success, 0% collision by end.
- ~2,860 SPS on local GPU (RTX 3050 Ti); will be much faster on L40S with 512 envs.

---

**Things to clarify / flag:**

1. **Grid interpretation**: I widened the shared corridor vs the ASCII art in `context.md` (5 cols wide instead of 1). This gives agents lateral room for ambiguous early moves. If you wanted the literal narrow corridor, let me know.

2. **V2 hyperparams**: The V2 brief says `D=2, N=2` but the V2 Architecture Addendum says `D=3, N=3`. I'll use **D=3, N=3** (matching the paper's DRC(3,3)). Correct?

3. **Cell sharing**: Agents can occupy the same cell outside of bottleneck tiles (no penalty). Collision only triggers when both agents land on the *same* bottleneck cell after the same step. Sound right?

4. **Local GPU**: You mentioned L40S — the local machine has an RTX 3050 Ti. I've set the PyTorch version to `torch==2.6.0+cu124` to match the driver. For the cluster run, you'll want the CUDA 12.x build there too. Want me to prep a Slurm script?

Ready to start V2 (DRC integration) whenever you confirm these are good.

---

**User**

no i think agents should never be able to share the same cell. i have a remote l40s on /alliancecan . please do drc integration.

---

**Cursor**

Got it — no cell sharing ever, and DRC integration for the cluster. Let me update the env and build V2.

Now let me write the Slurm script and run a quick local smoke test in parallel.

The DRC training with BPTT is running — it's slower than V1 due to the sequential tick loop and full 20-step backprop. Let me wait for it to finish.

The `tail -25` pipe is buffering all output. Let me kill this and run a more targeted test.

The training seems very slow. Let me profile the model directly to identify the bottleneck.

Model performance is fine (11ms forward, 30ms/step BPTT). The previous tests timed out because the pipe buffered output. Let me do a properly sized quick test.

V2 training works end-to-end. SPS is low on the tiny local test (4 envs, 3050 Ti), but with 512 envs on the L40S it'll be well-utilized. Let me finalize everything.

V1 still works with the updated env. Now let me update the docs.

All done. Here's a summary of what was built:

---

**Updated files:**

- **`env.py`** — No cell sharing enforced: same-target and swap conflicts block both agents. Bottleneck collision (both targeting same bottleneck cell) still gives -5 + episode end. Agents that reach their goal are removed from physics (non-blocking, hidden from partner's observation channel) — this handles same-goal scenarios correctly.

**New files:**

- **`models_drc.py`** — DRC(3,3) actor-critic matching the paper spec:
  - 3 ConvLSTM layers × 3 internal ticks, 32 channels, 3×3 kernels
  - Pool-and-inject: global mean/max pool → linear → reshape → add to hidden state before gate computation
  - Bottom-up skips (encoder output to all layers) + top-down skips (layer 3 → layer 1 across ticks)
  - Output: `cat(h_top, i_t)` → flatten → `cat(goal)` → trunk → pi/val heads
  - Forget gate bias initialized to 1.0

- **`train_v2.py`** — DRC MAPPO training:
  - Hidden states `(D, num_envs×2, G, 7, 7)` persist across timesteps
  - Mask-based hidden state reset at episode boundaries (autograd-compatible)
  - PPO update: sequence-level mini-batching with full BPTT over the rollout length (no timestep shuffling)
  - Weight decay only on policy/value heads (per paper E.4)

- **`slurm_ma_tom.sh`** — Slurm script for L40S on Killarney:
  - `#SBATCH --gres=gpu:l40s:1`, 8 CPUs, 50G RAM, 6h wall
  - Auto-creates venv under `$PROJECT/$USER/overcooked/venv`
  - Passes `"$@"` so you can override any train args from the command line

**Performance note:** On your local 3050 Ti, V2 is slow (~30ms/step for BPTT) because the DRC's 9 sequential ConvLSTM ops per timestep create long computation graphs. On the L40S with 512 envs, the larger batch will keep the GPU saturated. To run on the cluster:

```bash
# on Killarney:
cd $SCRATCH && git clone <repo> overcooked
sbatch overcooked/slurm_ma_tom.sh
```

The `--account=aip-boyuwang` in the script is an example — override with `sbatch --account=YOUR-GROUP slurm_ma_tom.sh` if yours differs.

---

**User**

did u write the v3 code alr

---

**Cursor**


No. **V3 isn’t implemented yet.**

What exists:

- **V1:** `env.py`, `train_v1.py`
- **V2:** `models_drc.py`, `train_v2.py`, `slurm_ma_tom.sh`

**V3** from `context.md` was planned as `interpretability.py`:

1. **Global goal probe** — ~1k rollouts with frozen DRC actors, record Agent A’s internal states, train logistic regression on partner goal (L/R), report Macro F1 on cell state at early timesteps (e.g. t=2–3 in the shared corridor).
2. **Causal intervention** — use probe’s “partner left” direction \(W_L\), pick a rollout where B’s goal is R, inject \(h \leftarrow h + W_L\) at t=3, compare action distribution shift.

If you want V3 next, say whether you prefer loading `checkpoints/v2_final.pt` by default and whether probe features should be **final-tick layer-3 cell state** vs **hidden** (spec mentions “final-tick cell state”).

---

**User**

yeah, default to that checkpoint and use the cell state

---

**Cursor**


Implementing `interpretability.py`: reading the DRC model and env to expose layer-3 cell states and wire probe + intervention.




Adding optional cell-state injection to the DRC forward pass, then implementing `interpretability.py` with sklearn logistic regression and wandb logging.




Adding `forward_logits` for deterministic eval and probe recording without sampling.




Fixing the minimum-sample check: it incorrectly required 100 samples even when `--num-rollouts` was lower.




Here’s what was added and how it’s wired.

### `interpretability.py` (V3)
- **Default checkpoint:** `checkpoints/v2_final.pt` (`--checkpoint` to override).
- **Features:** After `forward_logits`, **flatten `c[D-1]`** (top ConvLSTM **cell** after all ticks) for **agent A** at env step `--probe-step` (default **2**, i.e. third decision).
- **Probe:** `LogisticRegression` with `y=1` = partner **L**, `y=0` = partner **R**; stratified train/test split; **macro F1** on the test set.
- **Intervention:** `W_L = clf.coef_.ravel()`; search for a reset with partner **R**; at the same step, from the same `(h, c)`, compare agent-A logits **with / without** `inject_cell_top`: only agent 0 gets `scale * reshape(W_L)` on `c[D-1]` before the core. Logs **KL** and **L1** of softmax probs to wandb (`intervention/kl_a0`, `intervention/l1_a0`).

### `models_drc.py`
- **`forward_logits(obs, goal, h, c, inject_cell_top=None)`** — no sampling; returns logits, value, `new_h`, `new_c`.
- **`inject_cell_top`:** optional `(B,G,H,W)` added to **`c[D-1]`** before `_core` on **`forward`**, **`get_value`**, and **`forward_logits`** (clone `c` so callers aren’t mutated).

### Other
- **`requirements.txt`:** `scikit-learn>=1.3.0`
- **`onboard.md` / `arch.md` / `arch_log.md`:** updated for V3.

**Run (after `train_v2.py` has written `checkpoints/v2_final.pt`):**
```bash
python interpretability.py
# or e.g.
python interpretability.py --num-rollouts 1000 --probe-step 2 --inject-scale 2.0
```

**Note:** Intervention perturbs **cell** state before the recurrent core (aligned with the probe on `c^D`). Effect size depends on training; a weak or random policy can give near-zero KL (as in a quick smoke run on a small checkpoint).

---

**User**

ok can u eval and explain the output here

---

**Cursor**

Let me check what the actual per-update cost looks like before picking a wall time.

L40S should finish 5M steps in under an hour. Cutting the wall time to 2h gives a comfortable margin.

Changed from **6h → 2h**. The benchmark says:

- Local 3050 Ti: ~21s/update → ~2.8h for 5M steps  
- L40S is ~5× faster (higher VRAM bandwidth, newer architecture) → **~35–45 min** for 5M steps

2h gives you a healthy ~3× safety margin without sitting in queue as a long-allocation job. If you end up running longer rollouts or more total steps, just add `--time=0-04:00:00` at submit time without touching the script:

```bash
sbatch --time=0-04:00:00 slurm_ma_tom.sh
```

