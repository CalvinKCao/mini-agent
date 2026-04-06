"""V3 interpretability: linear probe, ablation, negative controls,
corridor-specific causal interventions, and time-of-emergence analysis.

Scientific claim (narrow):
  Does agent A's DRC cell state at an early timestep (before corridor
  commitment) linearly encode agent B's hidden corridor intent?
  Does injecting the learned concept vector causally shift A's corridor
  choice toward the *specific* injected corridor?

Runbook
-------
  # 1. Train
  python train.py [--num-envs 512 --total-timesteps 5000000]

  # 2. Run all interpretability tests (fresh rollouts)
  python interpretability.py --checkpoint checkpoints/v3_final.pt

  # 3. Save traces first, then load (faster for multiple runs)
  python train.py --eval-only --checkpoint checkpoints/v3_final.pt \\
                  --traces-out traces/v3_eval.npz --num-eval-eps 1000
  python interpretability.py --traces-file traces/v3_eval.npz

  # 4. Run specific mode only
  python interpretability.py --checkpoint checkpoints/v3_final.pt --mode probe
  python interpretability.py --checkpoint checkpoints/v3_final.pt --mode ablation
  python interpretability.py --checkpoint checkpoints/v3_final.pt --mode negcontrol
  python interpretability.py --checkpoint checkpoints/v3_final.pt --mode intervention
  python interpretability.py --checkpoint checkpoints/v3_final.pt --mode emergence

Modes: all (default), probe, ablation, negcontrol, intervention, emergence
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from env import GRID_H, GRID_W, N_CORRIDORS, CORRIDOR_NAMES, ThreeCorridorVec
from models_drc import DRCActorCritic


# ── Loading ────────────────────────────────────────────────────────────
def load_model(ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    saved = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    G = saved.get("drc_channels", 32)
    D = saved.get("drc_depth",    3)
    N = saved.get("drc_ticks",    3)
    model = DRCActorCritic(
        obs_ch=3, goal_dim=N_CORRIDORS, G=G, D=D, N=N, H=GRID_H, W=GRID_W,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, G, D, N


# ── Data collection ────────────────────────────────────────────────────
def collect_rollouts(
    model, device, n_rollouts, max_steps, probe_steps,
    mask_partner=False, seed=42,
):
    """Run n_rollouts greedy episodes; record cell states + metadata.

    Returns
    -------
    cells       : dict {step: np.array (N, G*H*W)}  — layer-3 cell for agent A
    intents     : np.array (N,)  int — B's corridor intent (0/1/2)
    corridors_a : np.array (N,)  int — A's committed corridor (-1 = none)
    corridors_b : np.array (N,)  int — B's committed corridor
    h_at_probe  : dict {step: list of (D, G, H, W) tensors} — full hidden for interventions
    c_at_probe  : dict {step: list of (D, G, H, W) tensors}
    obs_at_probe: dict {step: list of (2, 3, H, W) np arrays}
    goal_at_probe: dict {step: list of (2, N_COR) np arrays}
    """
    env = ThreeCorridorVec(num_envs=1, max_steps=max_steps,
                           seed=seed, mask_partner_obs=mask_partner)
    rng = np.random.default_rng(seed)

    cells        = {s: [] for s in probe_steps}
    hs_at_probe  = {s: [] for s in probe_steps}
    cs_at_probe  = {s: [] for s in probe_steps}
    obs_at_probe = {s: [] for s in probe_steps}
    goa_at_probe = {s: [] for s in probe_steps}
    intents, corridors_a, corridors_b = [], [], []

    max_tries = n_rollouts * 8
    n_done, tries = 0, 0
    max_probe = max(probe_steps)

    # primary probe step must be reached; others are best-effort (for emergence)
    primary_step = min(probe_steps)

    while n_done < n_rollouts and tries < max_tries:
        tries += 1
        env.rng = np.random.default_rng(int(rng.integers(1 << 30)))
        obs_np, goal_np = env.reset()
        intent = int(env.goals[0, 1])
        h, c = model.initial_state(2, device=device)

        ep_cells = {}
        ep_hs, ep_cs, ep_obs, ep_goa = {}, {}, {}, {}
        ca, cb = -1, -1

        for s in range(max_probe + 1):
            obs_t  = torch.from_numpy(obs_np).reshape(2, 3, GRID_H, GRID_W).to(device)
            goal_t = torch.from_numpy(goal_np).reshape(2, N_CORRIDORS).to(device)

            with torch.no_grad():
                logits, _, h, c = model.forward_logits(obs_t, goal_t, h, c)

            if s in probe_steps:
                ep_cells[s] = c[-1, 0].cpu().numpy().flatten().astype(np.float32)
                ep_hs[s]    = h.cpu()
                ep_cs[s]    = c.cpu()
                ep_obs[s]   = obs_np.copy()
                ep_goa[s]   = goal_np.copy()

            acts    = logits.argmax(-1).cpu().numpy()
            obs_np, goal_np, _, done, info = env.step(acts.reshape(1, 2))

            if done[0]:
                ca = int(info["chosen_corridor_a"][0])
                cb = int(info["chosen_corridor_b"][0])
                break

        # require only that the primary (smallest) probe step was reached
        if primary_step not in ep_cells:
            continue

        for s in probe_steps:
            if s in ep_cells:
                cells[s].append(ep_cells[s])
                hs_at_probe[s].append(ep_hs[s])
                cs_at_probe[s].append(ep_cs[s])
                obs_at_probe[s].append(ep_obs[s])
                goa_at_probe[s].append(ep_goa[s])
        intents.append(intent)
        corridors_a.append(ca)
        corridors_b.append(cb)
        n_done += 1

    for s in probe_steps:
        if cells[s]:
            feat_dim = cells[s][0].shape[0]
            cells[s] = np.stack(cells[s])
        else:
            feat_dim = model.G * GRID_H * GRID_W
            cells[s] = np.zeros((0, feat_dim))
    intents    = np.array(intents,    dtype=np.int32)
    corridors_a = np.array(corridors_a, dtype=np.int32)
    corridors_b = np.array(corridors_b, dtype=np.int32)

    return     cells, intents, corridors_a, corridors_b, hs_at_probe, cs_at_probe, \
           obs_at_probe, goa_at_probe, model.G


def _probe(X, y, seed=0):
    """Fit + evaluate logistic regression. Returns (clf, macro_f1, cm, f1_per_class)."""
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y,
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    macro_f1      = f1_score(y_te, pred, average="macro",    zero_division=0)
    f1_per_class  = f1_score(y_te, pred, average=None,       zero_division=0)
    cm            = confusion_matrix(y_te, pred, labels=list(range(N_CORRIDORS)))
    return clf, macro_f1, cm, f1_per_class


# ── A. Probe ───────────────────────────────────────────────────────────
def run_probe(cells, intents, probe_step, step=None, log=True):
    """Train 3-class linear probe on c^D at probe_step.  Returns clf."""
    X = cells[probe_step]
    y = intents
    if len(X) < 10:
        print(f"[PROBE @ step {probe_step}] too few samples ({len(X)}) — skipping")
        return None, 0.0
    clf, f1, cm, f1_cls = _probe(X, y)

    chance = 1.0 / N_CORRIDORS
    print(f"\n[PROBE @ step {probe_step}]  macro F1={f1:.4f}  (chance={chance:.3f})")
    for k, name in enumerate(CORRIDOR_NAMES):
        print(f"  {name:6s}: F1={f1_cls[k]:.4f}")
    print("  Confusion matrix (rows=true, cols=pred):")
    for row, name in zip(cm, CORRIDOR_NAMES):
        print(f"    {name:6s}: {row}")

    if log and step is not None:
        cm_table = wandb.Table(
            columns=["true \\ pred"] + CORRIDOR_NAMES,
            data=[[CORRIDOR_NAMES[i]] + list(cm[i]) for i in range(N_CORRIDORS)],
        )
        wandb.log({
            "probe/macro_f1":     f1,
            "probe/chance":       chance,
            "probe/f1_left":      f1_cls[0],
            "probe/f1_center":    f1_cls[1],
            "probe/f1_right":     f1_cls[2],
            "probe/confusion_matrix": cm_table,
            "probe/n_samples":    len(y),
        }, step=step)
    return clf, f1


# ── B. Ablation: partner visibility ────────────────────────────────────
def run_ablation(model, device, n_rollouts, max_steps, probe_step, seed, log_step=None):
    """Re-probe after zeroing the partner-position channel in obs.

    Expected: F1 drops materially vs full observation.
    """
    cells_m, intents_m, _, _, _, _, _, _, _ = collect_rollouts(
        model, device, n_rollouts, max_steps, [probe_step],
        mask_partner=True, seed=seed + 1000,
    )
    if len(intents_m) < 20:
        print("[ABLATION] too few samples — skipping")
        return None
    _, f1_masked, cm_m, f1_cls_m = _probe(cells_m[probe_step], intents_m)
    print(f"\n[ABLATION @ step {probe_step}] masked F1={f1_masked:.4f}  "
          f"(expected < full-obs F1)")
    if log_step is not None:
        wandb.log({
            "ablation/macro_f1_masked": f1_masked,
            "ablation/f1_left_masked":  f1_cls_m[0],
            "ablation/f1_center_masked":f1_cls_m[1],
            "ablation/f1_right_masked": f1_cls_m[2],
        }, step=log_step)
    return f1_masked


# ── C. Negative controls ───────────────────────────────────────────────
def run_neg_control(cells, intents, probe_step, n_shuffles=5, seed=0, log_step=None):
    """Shuffle partner-intent labels; expected F1 → chance (≈0.33).
    Also tests random-direction interventions in caller.
    """
    X = cells[probe_step]
    y = intents
    shuffled_f1s = []
    rng = np.random.default_rng(seed)
    for _ in range(n_shuffles):
        y_sh = rng.permutation(y)
        _, f1_sh, _, _ = _probe(X, y_sh, seed=int(rng.integers(1 << 20)))
        shuffled_f1s.append(f1_sh)
    mean_shuf = float(np.mean(shuffled_f1s))
    print(f"\n[NEG CONTROL] shuffled-label F1 (avg over {n_shuffles}): {mean_shuf:.4f} "
          f"(chance=0.33)")
    if log_step is not None:
        wandb.log({
            "negcontrol/shuffled_f1_mean": mean_shuf,
            "negcontrol/shuffled_f1_std":  float(np.std(shuffled_f1s)),
        }, step=log_step)
    return mean_shuf


# ── D. Corridor-specific causal interventions ──────────────────────────
def _run_episode_with_injection(model, device, obs0, goal0, h0, c0,
                                 inject_arr, max_steps, inject_step,
                                 G, env_state_seed, env):
    """Re-run one episode from the start; at inject_step inject into c^D.

    Returns (corridor_committed_a, action_dist_change_l1)
    """
    env.rng = np.random.default_rng(env_state_seed)
    obs_np, goal_np = env.reset()
    h, c = model.initial_state(2, device=device)

    p0_saved = None
    corridor_a = -1

    for s in range(max_steps + 1):
        obs_t  = torch.from_numpy(obs_np).reshape(2, 3, GRID_H, GRID_W).to(device)
        goal_t = torch.from_numpy(goal_np).reshape(2, N_CORRIDORS).to(device)

        inject = None
        if s == inject_step and inject_arr is not None:
            inject = torch.from_numpy(inject_arr).to(device)

        with torch.no_grad():
            logits, _, h, c = model.forward_logits(obs_t, goal_t, h, c,
                                                    inject_cell_top=inject)
        if s == inject_step:
            p0_saved = F.softmax(logits[0], dim=-1).cpu()

        acts = logits.argmax(-1).cpu().numpy()
        obs_np, goal_np, _, done, info = env.step(acts.reshape(1, 2))
        if done[0]:
            corridor_a = int(info["chosen_corridor_a"][0])
            break

    if corridor_a == -1:
        corridor_a = int(env.corridor_committed[0, 0])
    return corridor_a, p0_saved


def run_interventions(
    model, device, G,
    clf,                         # fitted probe
    probe_step,
    n_trials=100,
    scale=1.0,
    max_steps=30,
    seed=42,
    log_step=None,
):
    """For each corridor k: inject W_k, measure A's corridor choice.

    Tests:
      1. Targeted injection (W_k per class) → P(A→k) should rise
      2. Random injection of same norm      → should not rise (neg control)
    """
    rng = np.random.default_rng(seed)
    env = ThreeCorridorVec(num_envs=1, max_steps=max_steps, seed=seed + 5000)

    # concept vectors: coef_[k] has shape (G*H*W,) in OvR
    W = clf.coef_.astype(np.float32)    # (N_CORRIDORS, feat_dim)
    W_norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9

    print(f"\n[INTERVENTION @ step {probe_step}]  inject_scale={scale}")
    table_rows = []

    for k_tgt in range(N_CORRIDORS):
        W_k = (W[k_tgt] / W_norms[k_tgt]) * scale   # unit-normed × scale
        W_inject = np.zeros((2, G, GRID_H, GRID_W), dtype=np.float32)
        W_inject[0] = W_k.reshape(G, GRID_H, GRID_W)

        # random vector of same norm as W_inject[0] (neg control)
        rnd = rng.standard_normal(W_inject[0].shape).astype(np.float32)
        rnd /= (np.linalg.norm(rnd) + 1e-9) / (np.linalg.norm(W_inject[0]) + 1e-9)
        W_rand = np.zeros_like(W_inject)
        W_rand[0] = rnd

        counts_tgt  = np.zeros(N_CORRIDORS + 1, dtype=np.int32)   # idx 0-2 + -1
        counts_base = np.zeros(N_CORRIDORS + 1, dtype=np.int32)
        counts_rand = np.zeros(N_CORRIDORS + 1, dtype=np.int32)
        n_valid = 0

        for trial in range(n_trials * 3):
            if n_valid >= n_trials:
                break
            ep_seed = int(rng.integers(1 << 30))
            env.rng = np.random.default_rng(ep_seed)
            obs_np, goal_np = env.reset()
            # only keep rollouts where B's intent ≠ k_tgt (so injection is "counterfactual")
            if int(env.goals[0, 1]) == k_tgt:
                continue

            ca_tgt,  _ = _run_episode_with_injection(
                model, device, None, None, None, None,
                W_inject, max_steps, probe_step, G, ep_seed, env,
            )
            ca_base, _ = _run_episode_with_injection(
                model, device, None, None, None, None,
                None, max_steps, probe_step, G, ep_seed, env,
            )
            ca_rand, _ = _run_episode_with_injection(
                model, device, None, None, None, None,
                W_rand, max_steps, probe_step, G, ep_seed, env,
            )

            counts_tgt[ca_tgt + 1]  += 1
            counts_base[ca_base + 1] += 1
            counts_rand[ca_rand + 1] += 1
            n_valid += 1

        p_tgt_tgt  = counts_tgt[k_tgt + 1]  / max(n_valid, 1)
        p_base_tgt = counts_base[k_tgt + 1] / max(n_valid, 1)
        p_rand_tgt = counts_rand[k_tgt + 1] / max(n_valid, 1)
        delta      = p_tgt_tgt - p_base_tgt

        row = [CORRIDOR_NAMES[k_tgt], f"{p_base_tgt:.3f}", f"{p_tgt_tgt:.3f}",
               f"{p_rand_tgt:.3f}", f"{delta:+.3f}", n_valid]
        table_rows.append(row)
        print(f"  inject→{CORRIDOR_NAMES[k_tgt]:6s}: "
              f"base={p_base_tgt:.3f}  injected={p_tgt_tgt:.3f}  "
              f"random={p_rand_tgt:.3f}  Δ={delta:+.3f}  (n={n_valid})")

    if log_step is not None:
        tbl = wandb.Table(
            columns=["inject_corridor", "P(A=k|base)", "P(A=k|inject_Wk)",
                     "P(A=k|rand)", "delta", "n_trials"],
            data=table_rows,
        )
        wandb.log({"intervention/corridor_table": tbl}, step=log_step)


# ── E. Time-of-emergence ───────────────────────────────────────────────
def run_emergence(cells, intents, probe_steps, log_step=None):
    """Probe macro F1 at each probe step to see when representation emerges."""
    print(f"\n[EMERGENCE]  probe steps: {probe_steps}")
    f1s = {}
    for s in sorted(probe_steps):
        X = cells.get(s, np.zeros((0, 1)))
        if len(X) < 10:
            print(f"  step {s}: too few samples ({len(X)}) — skipping")
            continue
        _, f1, _, _ = _probe(X, intents, seed=s)
        f1s[s] = f1
        print(f"  step {s}: macro F1 = {f1:.4f}")

    if log_step is not None:
        for s, f1 in f1s.items():
            wandb.log({f"emergence/f1_step{s}": f1}, step=log_step)
    return f1s


# ── F. Held-out generalization (split by episode seed parity) ──────────
def run_held_out(cells, intents, probe_step, log_step=None):
    """Train on first half, test on second half (simulates held-out layouts)."""
    X = cells[probe_step]
    y = intents
    n = len(y)
    half = n // 2
    X_tr, X_te = X[:half], X[half:]
    y_tr, y_te = y[:half], y[half:]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        print("[HELD-OUT] insufficient class diversity — skipping")
        return None
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    f1 = f1_score(y_te, pred, average="macro", zero_division=0)
    print(f"\n[HELD-OUT]  held-out half macro F1 = {f1:.4f}")
    if log_step is not None:
        wandb.log({"heldout/macro_f1": f1}, step=log_step)
    return f1


# ── main ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   type=str, default="checkpoints/v3_final.pt")
    p.add_argument("--traces-file",  type=str, default=None,
                   help="load pre-saved traces instead of re-running rollouts")
    p.add_argument("--num-rollouts", type=int, default=1000)
    p.add_argument("--probe-step",   type=int, default=2,
                   help="primary probe timestep (default 2)")
    p.add_argument("--max-steps",    type=int, default=30)
    p.add_argument("--inject-scale", type=float, default=1.0)
    p.add_argument("--n-intervention-trials", type=int, default=150)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--wandb-project",type=str, default="forked-corridor-tom")
    p.add_argument("--run-name",     type=str, default=None)
    p.add_argument("--mode",         type=str, default="all",
                   choices=["all", "probe", "ablation", "negcontrol",
                            "intervention", "emergence"])
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isfile(args.checkpoint) and args.traces_file is None:
        raise FileNotFoundError(
            f"Checkpoint {args.checkpoint} not found. "
            "Train with train.py first, or pass --traces-file."
        )

    run_name = args.run_name or f"v3-interp-step{args.probe_step}-s{args.seed}"
    wandb.init(
        project=args.wandb_project, name=run_name,
        config={**vars(args), "test": "v3-interpretability"},
    )

    model, G, D, N = load_model(args.checkpoint, device)
    print(f"Loaded DRC({D},{N}) G={G} from {args.checkpoint}")

    # ── collect data ─────────────────────────────────────────────────
    probe_steps = [0, 1, 2, 3, 4, 5]

    if args.traces_file and os.path.isfile(args.traces_file):
        print(f"Loading traces from {args.traces_file}")
        tr = np.load(args.traces_file)
        cells = {int(k.replace("cell_s", "")): tr[k] for k in tr.files
                 if k.startswith("cell_s")}
        intents    = tr["partner_intents"]
        corridors_a = tr["corridors_a"]
        corridors_b = np.zeros_like(corridors_a) - 1
        # traces don't contain full hidden states — interventions will re-run
        hs_at_probe = {s: [] for s in probe_steps}
        cs_at_probe = {s: [] for s in probe_steps}
        obs_at_probe = {s: [] for s in probe_steps}
        goa_at_probe = {s: [] for s in probe_steps}
    else:
        print(f"Collecting {args.num_rollouts} rollouts (probe steps {probe_steps})...")
        (cells, intents, corridors_a, corridors_b,
         hs_at_probe, cs_at_probe, obs_at_probe, goa_at_probe, _G) = collect_rollouts(
            model, device, args.num_rollouts, args.max_steps, probe_steps,
            mask_partner=False, seed=args.seed,
        )
        print(f"  collected: {len(intents)}  "
              f"A committed: {(corridors_a >= 0).sum()}/{len(corridors_a)}")

    gstep = 0
    do = lambda mode: args.mode in ("all", mode)

    # ── A. Probe ──────────────────────────────────────────────────────
    clf_main = None
    if do("probe"):
        clf_main, f1_main = run_probe(
            cells, intents, probe_step=args.probe_step, step=gstep, log=True,
        )
        # held-out generalization
        run_held_out(cells, intents, probe_step=args.probe_step, log_step=gstep)
        gstep += 1

    # ── B. Ablation: mask partner channel ────────────────────────────
    if do("ablation"):
        f1_masked = run_ablation(
            model, device, min(args.num_rollouts, 400), args.max_steps,
            args.probe_step, seed=args.seed, log_step=gstep,
        )
        if f1_masked is not None and clf_main is not None:
            delta = f1_main - f1_masked
            print(f"  ΔF1 (full - masked) = {delta:+.4f}  "
                  f"({'✓ partner info used' if delta > 0.05 else '✗ weak effect'})")
            wandb.log({"ablation/delta_f1": delta}, step=gstep)
        gstep += 1

    # ── C. Negative controls ─────────────────────────────────────────
    if do("negcontrol"):
        run_neg_control(cells, intents, args.probe_step, n_shuffles=5,
                        seed=args.seed, log_step=gstep)
        gstep += 1

    # ── D. Corridor-specific interventions ───────────────────────────
    if do("intervention"):
        # need a probe for concept vectors
        if clf_main is None:
            clf_main, _ = run_probe(
                cells, intents, probe_step=args.probe_step, step=None, log=False,
            )
        run_interventions(
            model, device, G,
            clf=clf_main,
            probe_step=args.probe_step,
            n_trials=args.n_intervention_trials,
            scale=args.inject_scale,
            max_steps=args.max_steps,
            seed=args.seed,
            log_step=gstep,
        )
        gstep += 1

    # ── E. Time-of-emergence ─────────────────────────────────────────
    if do("emergence"):
        run_emergence(cells, intents, probe_steps=probe_steps, log_step=gstep)
        gstep += 1

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
