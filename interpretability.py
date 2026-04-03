"""V3 — linear probe on layer-3 cell state + causal cell-state intervention.

Default checkpoint: checkpoints/v2_final.pt
Features: flattened top-layer ConvLSTM **cell** state c^D after N ticks.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from env import ForkedCorridorVec
from models_drc import DRCActorCritic


def load_drc(ckpt_path, device):
    try:
        extra = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        extra = torch.load(ckpt_path, map_location=device)
    state = extra["model"] if isinstance(extra, dict) and "model" in extra else extra
    args = extra.get("args", {}) if isinstance(extra, dict) else {}
    G = args.get("drc_channels", 32)
    D = args.get("drc_depth", 3)
    N = args.get("drc_ticks", 3)
    model = DRCActorCritic(obs_ch=3, goal_dim=2, G=G, D=D, N=N).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, G, D, N


def rollout_features(
    env, model, device, probe_step, max_steps, greedy=True,
):
    """One episode on num_envs=1. Returns (feature_vec or None, partner_goal_R_bool, goals_row).

    partner_goal: 0=L, 1=R for agent B.
    Feature is c_top for agent 0 after forward at env step index `probe_step`, if episode alive.
    """
    obs_np, goal_np = env.reset()
    h, c = model.initial_state(2, device=device)

    for s in range(max_steps + 1):
        obs = torch.from_numpy(obs_np).reshape(2, 3, 7, 7).to(device)
        goal = torch.from_numpy(goal_np).reshape(2, 2).to(device)

        logits, _val, h, c = model.forward_logits(obs, goal, h, c)
        if greedy:
            actions = logits.argmax(dim=-1).cpu().numpy()
        else:
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy()

        if s == probe_step:
            c_top = c[-1, 0].detach().flatten().cpu().numpy().astype(np.float64)
            partner = int(env.goals[0, 1])
            goals_row = env.goals[0].copy()
            return c_top, partner, goals_row

        obs_np, goal_np, _rew, done, _info = env.step(actions.reshape(1, 2))
        if done[0]:
            return None, None, None

    return None, None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/v2_final.pt")
    p.add_argument("--num-rollouts", type=int, default=1000)
    p.add_argument("--probe-step", type=int, default=2,
                   help="0-based env step index at which to record c^D (default 2 ≈ t=3)")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--inject-scale", type=float, default=1.0,
                   help="scale for W_L added to c^D before core (agent A only)")
    p.add_argument("--wandb-project", type=str, default="forked-corridor-tom")
    p.add_argument("--run-name", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"checkpoint not found: {args.checkpoint}\n"
            "Train with train_v2.py first or pass --checkpoint."
        )

    run_name = args.run_name or f"v3-probe-step{args.probe_step}-s{args.seed}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            **vars(args),
            "architecture": "V3: linear probe on layer-3 cell state + cell injection",
        },
    )

    model, G, D, _N = load_drc(args.checkpoint, device)

    env = ForkedCorridorVec(num_envs=1, max_steps=args.max_steps, seed=int(rng.integers(1 << 30)))

    X_list, y_list = [], []
    attempts = 0
    max_attempts = args.num_rollouts * 5
    while len(X_list) < args.num_rollouts and attempts < max_attempts:
        attempts += 1
        env.rng = np.random.default_rng(int(rng.integers(1 << 30)))
        feat, partner_goal, _g = rollout_features(
            env, model, device, args.probe_step, args.max_steps,
        )
        if feat is None:
            continue
        # y=1 iff partner's goal is Left (class 0 in env encoding)
        y_list.append(1 if partner_goal == 0 else 0)
        X_list.append(feat)

    if len(X_list) < args.num_rollouts:
        raise RuntimeError(
            f"only collected {len(X_list)}/{args.num_rollouts} valid probe samples "
            f"(episodes often end before probe_step={args.probe_step}; "
            f"try lowering --probe-step or increasing --max-steps)"
        )

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed,
        )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)

    wandb.log({
        "probe/macro_f1": macro_f1,
        "probe/n_samples": len(X_list),
        "probe/pos_rate": float(y.mean()),
    })
    print(f"probe samples: {len(X_list)}  macro F1 (test): {macro_f1:.4f}")

    # W_L: weight vector increasing P(partner Left); sklearn y=1 is Left
    W_L = clf.coef_.astype(np.float64).ravel()

    # --- inception-style intervention: partner R, inject +W_L into A's c^D ---
    inj_kl, inj_l1, found = None, None, False
    tries = 0
    while tries < 5000 and not found:
        tries += 1
        env.rng = np.random.default_rng(int(rng.integers(1 << 30)))
        obs_np, goal_np = env.reset()
        if env.goals[0, 1] != 1:
            continue
        h, c = model.initial_state(2, device=device)

        for s in range(args.max_steps + 1):
            obs = torch.from_numpy(obs_np).reshape(2, 3, 7, 7).to(device)
            goal = torch.from_numpy(goal_np).reshape(2, 2).to(device)

            if s == args.probe_step:
                inject = torch.zeros(2, G, 7, 7, device=device)
                inject[0] = torch.from_numpy(
                    (W_L * args.inject_scale).reshape(G, 7, 7).astype(np.float32)
                ).to(device)

                with torch.no_grad():
                    lb0, _, _, _ = model.forward_logits(obs, goal, h, c)
                    lb1, _, _, _ = model.forward_logits(
                        obs, goal, h, c, inject_cell_top=inject,
                    )
                p0 = F.softmax(lb0[0], dim=-1)
                p1 = F.softmax(lb1[0], dim=-1)
                log_p0 = F.log_softmax(lb0[0], dim=-1)
                log_p1 = F.log_softmax(lb1[0], dim=-1)
                inj_kl = float((p0 * (log_p0 - log_p1)).sum().item())
                inj_l1 = float((p0 - p1).abs().sum().item())
                found = True
                break

            logits, _v, h, c = model.forward_logits(obs, goal, h, c)
            actions = logits.argmax(dim=-1).cpu().numpy()
            obs_np, goal_np, _rew, done, _info = env.step(actions.reshape(1, 2))
            if done[0]:
                break

    if found:
        wandb.log({
            "intervention/kl_a0": inj_kl,
            "intervention/l1_a0": inj_l1,
            "intervention/found_partner_R": 1,
        })
        print(f"intervention (partner=R @ step {args.probe_step}): KL={inj_kl:.5f}  L1(p)={inj_l1:.5f}")
    else:
        wandb.log({"intervention/found_partner_R": 0})
        print("intervention: no suitable rollout (partner=R, survived to probe step)")

    wandb.finish()


if __name__ == "__main__":
    main()
