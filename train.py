"""V3 — DRC(3,3) MAPPO on the three-corridor coordination environment.

Runbook
-------
  Train:
    python train.py [--num-envs 512 --total-timesteps 5000000]
    python train.py --device cuda --run-name v3-drc33-s1

  Resume:
    python train.py --resume checkpoints/v3_step1000000.pt

  Evaluate + save rollout traces (for interpretability.py):
    python train.py --eval-only --checkpoint checkpoints/v3_final.pt \\
                    --traces-out traces/v3_eval.npz --num-eval-eps 500
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import wandb
from compile_safe import maybe_compile
from env import GRID_H, GRID_W, N_CORRIDORS, ThreeCorridorVec
from models_drc import DRCActorCritic


# ── args ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",            type=int,   default=1)
    p.add_argument("--total-timesteps", type=int,   default=5_000_000)
    p.add_argument("--num-envs",        type=int,   default=512)
    p.add_argument("--num-steps",       type=int,   default=20)
    p.add_argument("--num-minibatches", type=int,   default=4)
    p.add_argument("--update-epochs",   type=int,   default=4)
    p.add_argument("--lr",              type=float, default=4e-4)
    p.add_argument("--gamma",           type=float, default=0.97)
    p.add_argument("--gae-lambda",      type=float, default=0.95)
    p.add_argument("--clip-coef",       type=float, default=0.2)
    p.add_argument("--ent-coef",        type=float, default=0.01)
    p.add_argument("--vf-coef",         type=float, default=0.5)
    p.add_argument("--max-grad-norm",   type=float, default=0.5)
    p.add_argument("--weight-decay",    type=float, default=1e-5)
    p.add_argument("--max-steps",       type=int,   default=30)
    p.add_argument("--drc-depth",       type=int,   default=3)
    p.add_argument("--drc-ticks",       type=int,   default=3)
    p.add_argument("--drc-channels",    type=int,   default=32)
    p.add_argument("--wandb-project",   type=str,   default="forked-corridor-tom")
    p.add_argument("--run-name",        type=str,   default=None)
    p.add_argument("--device",          type=str,   default="cuda")
    p.add_argument("--compile",         action="store_true", default=True)
    p.add_argument("--no-compile",      dest="compile", action="store_false")
    p.add_argument("--save-dir",        type=str,   default="checkpoints")
    p.add_argument("--resume",          type=str,   default=None,
                   help="checkpoint to resume from")
    # eval-only mode
    p.add_argument("--eval-only",       action="store_true")
    p.add_argument("--checkpoint",      type=str,   default=None)
    p.add_argument("--traces-out",      type=str,   default=None,
                   help="save eval rollout traces to this npz path")
    p.add_argument("--num-eval-eps",    type=int,   default=500)
    return p.parse_args()


# ── model helpers ─────────────────────────────────────────────────────
def build_model(args, device):
    model = DRCActorCritic(
        obs_ch=3, goal_dim=N_CORRIDORS,
        G=args.drc_channels, D=args.drc_depth, N=args.drc_ticks,
        H=GRID_H, W=GRID_W,
    ).to(device)
    return model


def load_checkpoint(path, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    saved_args = ckpt.get("args", {})
    return state, saved_args


# ── eval-only mode ────────────────────────────────────────────────────
def run_eval(args, device):
    state, saved_args = load_checkpoint(args.checkpoint, device)
    G  = saved_args.get("drc_channels", args.drc_channels)
    D  = saved_args.get("drc_depth",    args.drc_depth)
    N_ = saved_args.get("drc_ticks",    args.drc_ticks)
    model = DRCActorCritic(
        obs_ch=3, goal_dim=N_CORRIDORS, G=G, D=D, N=N_,
        H=GRID_H, W=GRID_W,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    env = ThreeCorridorVec(num_envs=1, max_steps=args.max_steps, seed=args.seed)

    corridor_a_counts = np.zeros(N_CORRIDORS + 1, dtype=np.int32)  # +1 for -1 (no commit)
    successes, collisions, ep_lens = [], [], []
    partner_intents, corridors_a, corridors_b = [], [], []

    # for trace saving: cell state at step 2 and 3
    probe_steps = [2, 3]
    trace_cells  = {s: [] for s in probe_steps}
    trace_labels = []
    trace_ca     = []

    n_done = 0
    rng = np.random.default_rng(args.seed)
    while n_done < args.num_eval_eps:
        env.rng = np.random.default_rng(int(rng.integers(1 << 30)))
        obs_np, goal_np = env.reset()
        pi = int(env.goals[0, 1])
        h, c = model.initial_state(2, device=device)
        ep_cells = {}

        for s in range(args.max_steps + 1):
            obs_t  = torch.from_numpy(obs_np).reshape(2, 3, GRID_H, GRID_W).to(device)
            goal_t = torch.from_numpy(goal_np).reshape(2, N_CORRIDORS).to(device)

            with torch.no_grad():
                logits, _, h, c = model.forward_logits(obs_t, goal_t, h, c)

            if s in probe_steps:
                ep_cells[s] = c[-1, 0].cpu().numpy().flatten().astype(np.float32)

            acts = logits.argmax(-1).cpu().numpy()
            obs_np, goal_np, _, done, info = env.step(acts.reshape(1, 2))

            if done[0]:
                ca = int(info["chosen_corridor_a"][0])
                cb = int(info["chosen_corridor_b"][0])
                succ = bool(info["successes"][0])
                coll = bool(info["collisions"][0])
                elen = int(info["episode_lengths"][0])
                successes.append(succ)
                collisions.append(coll)
                ep_lens.append(elen)
                partner_intents.append(pi)
                corridors_a.append(ca)
                corridors_b.append(cb)
                corridor_a_counts[ca + 1] += 1
                if all(s in ep_cells for s in probe_steps):
                    for s in probe_steps:
                        trace_cells[s].append(ep_cells[s])
                    trace_labels.append(pi)
                    trace_ca.append(ca)
                n_done += 1
                break

    sr  = np.mean(successes)
    cr  = np.mean(collisions)
    el  = np.mean(ep_lens)
    print(f"\nEval over {n_done} episodes:")
    print(f"  success={sr:.3f}  collision={cr:.3f}  ep_len={el:.1f}")
    print(f"  A corridor distribution (L/C/R/none): {corridor_a_counts[1:]}/{corridor_a_counts[0]}")

    if args.traces_out and trace_labels:
        os.makedirs(os.path.dirname(args.traces_out) or ".", exist_ok=True)
        np.savez(
            args.traces_out,
            **{f"cell_s{s}": np.stack(trace_cells[s]) for s in probe_steps},
            partner_intents=np.array(trace_labels, dtype=np.int32),
            corridors_a=np.array(trace_ca, dtype=np.int32),
            G=np.array(G), D=np.array(D),
        )
        print(f"traces saved → {args.traces_out}  (n={len(trace_labels)})")

    return sr


# ── training ──────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        ckpt = args.checkpoint or os.path.join(args.save_dir, "v3_final.pt")
        args.checkpoint = ckpt
        run_eval(args, device)
        return

    run_name = args.run_name or f"v3-drc{args.drc_depth}{args.drc_ticks}-s{args.seed}"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_amp = device.type == "cuda"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            **vars(args),
            "env": "ThreeCorridorVec",
            "grid": f"{GRID_H}x{GRID_W}",
            "goal_dim": N_CORRIDORS,
            "architecture": (
                f"DRC({args.drc_depth},{args.drc_ticks}) G={args.drc_channels}, "
                "pool-and-inject, BU+TD skips, param-shared MAPPO"
            ),
        },
    )

    env = ThreeCorridorVec(num_envs=args.num_envs, max_steps=args.max_steps, seed=args.seed)
    model = build_model(args, device)
    model, cstat = maybe_compile(model, device=device, enabled=args.compile, mode="default")
    print(f"torch.compile status: {cstat}")
    wandb.config.update({"torch_compile_status": cstat}, allow_val_change=True)

    start_update = 1
    if args.resume:
        state, _ = load_checkpoint(args.resume, device)
        model.load_state_dict(state, strict=False)
        print(f"resumed from {args.resume}")

    head_params = {id(p) for n, p in model.named_parameters()
                   if "pi_head" in n or "val_head" in n}
    optimizer = torch.optim.Adam([
        {"params": [p for p in model.parameters() if id(p) not in head_params],
         "weight_decay": 0.0},
        {"params": [p for p in model.parameters() if id(p) in head_params],
         "weight_decay": args.weight_decay},
    ], lr=args.lr)

    n_agents       = args.num_envs * 2
    num_updates    = args.total_timesteps // (args.num_envs * args.num_steps)
    mb_seq_count   = n_agents // args.num_minibatches

    buf_obs  = torch.zeros(args.num_steps, args.num_envs, 2, 3, GRID_H, GRID_W)
    buf_goal = torch.zeros(args.num_steps, args.num_envs, 2, N_CORRIDORS)
    buf_act  = torch.zeros(args.num_steps, args.num_envs, 2, dtype=torch.long)
    buf_logp = torch.zeros(args.num_steps, args.num_envs, 2)
    buf_rew  = torch.zeros(args.num_steps, args.num_envs, 2)
    buf_done = torch.zeros(args.num_steps, args.num_envs)
    buf_val  = torch.zeros(args.num_steps, args.num_envs, 2)

    h, c = model.initial_state(n_agents, device=device)
    obs_np, goal_np = env.reset()
    next_obs  = torch.from_numpy(obs_np)
    next_goal = torch.from_numpy(goal_np)
    next_done = torch.zeros(args.num_envs)

    global_step = 0
    t0 = time.time()
    os.makedirs(args.save_dir, exist_ok=True)

    for update in range(start_update, num_updates + 1):
        frac   = 1.0 - (update - 1) / num_updates
        lr_now = frac * args.lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        init_h = h.detach().clone()
        init_c = c.detach().clone()

        # ── rollout ──────────────────────────────────────────────
        for step in range(args.num_steps):
            global_step += args.num_envs
            buf_obs[step]  = next_obs
            buf_goal[step] = next_goal
            buf_done[step] = next_done

            flat_obs  = next_obs.reshape(-1, 3, GRID_H, GRID_W).to(device)
            flat_goal = next_goal.reshape(-1, N_CORRIDORS).to(device)

            with torch.no_grad():
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    act, logp, _, val, h, c = model(flat_obs, flat_goal, h, c)

            act_cpu  = act.cpu().reshape(args.num_envs, 2)
            buf_act[step]  = act_cpu
            buf_logp[step] = logp.cpu().reshape(args.num_envs, 2)
            buf_val[step]  = val.cpu().reshape(args.num_envs, 2)

            obs_np, goal_np, rew_np, done_np, info = env.step(act_cpu.numpy())
            buf_rew[step]  = torch.from_numpy(rew_np)
            next_obs       = torch.from_numpy(obs_np)
            next_goal      = torch.from_numpy(goal_np)
            next_done      = torch.from_numpy(done_np.astype(np.float32))

            if done_np.any():
                dm = torch.from_numpy(done_np).to(device)
                mask = dm.unsqueeze(-1).expand(-1, 2).reshape(-1).bool()
                h[:, mask] = 0
                c[:, mask] = 0

            if "episode_returns" in info:
                ca = info["chosen_corridor_a"]
                pi = info["partner_intent"]
                wandb.log({
                    "charts/ep_return_mean": info["episode_returns"].mean(),
                    "charts/ep_length":      info["episode_lengths"].mean(),
                    "charts/success_rate":   info["successes"].mean(),
                    "charts/collision_rate": info["collisions"].mean(),
                    "charts/corridor_match": info["corridor_match"].mean(),
                    "charts/a_left_rate":    (ca == 0).mean(),
                    "charts/a_center_rate":  (ca == 1).mean(),
                    "charts/a_right_rate":   (ca == 2).mean(),
                    "charts/b_intent_left":  (pi == 0).mean(),
                    "charts/b_intent_right": (pi == 2).mean(),
                }, step=global_step)

        # ── GAE ──────────────────────────────────────────────────
        with torch.no_grad():
            nv_obs  = next_obs.reshape(-1, 3, GRID_H, GRID_W).to(device)
            nv_goal = next_goal.reshape(-1, N_CORRIDORS).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                next_val, h, c = model.get_value(nv_obs, nv_goal, h, c)
            next_val = next_val.cpu().reshape(args.num_envs, 2)

        advantages = torch.zeros_like(buf_rew)
        lastgae    = torch.zeros(args.num_envs, 2)
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nnt = (1.0 - next_done).unsqueeze(-1)
                nv  = next_val
            else:
                nnt = (1.0 - buf_done[t + 1]).unsqueeze(-1)
                nv  = buf_val[t + 1]
            delta   = buf_rew[t] + args.gamma * nv * nnt - buf_val[t]
            lastgae = delta + args.gamma * args.gae_lambda * nnt * lastgae
            advantages[t] = lastgae
        returns = advantages + buf_val

        b_obs  = buf_obs.reshape(args.num_steps, n_agents, 3, GRID_H, GRID_W)
        b_goal = buf_goal.reshape(args.num_steps, n_agents, N_CORRIDORS)
        b_act  = buf_act.reshape(args.num_steps, n_agents)
        b_logp = buf_logp.reshape(args.num_steps, n_agents)
        b_adv  = advantages.reshape(args.num_steps, n_agents)
        b_ret  = returns.reshape(args.num_steps, n_agents)
        b_val  = buf_val.reshape(args.num_steps, n_agents)
        b_done = buf_done.unsqueeze(-1).expand(-1, -1, 2).reshape(args.num_steps, n_agents)

        # ── PPO update ───────────────────────────────────────────
        clip_fracs = []
        for _epoch in range(args.update_epochs):
            perm = np.random.permutation(n_agents)
            for mb_start in range(0, n_agents, mb_seq_count):
                mb_idx = perm[mb_start: mb_start + mb_seq_count]
                mb_h = init_h[:, mb_idx].clone()
                mb_c = init_c[:, mb_idx].clone()
                all_logp, all_ent, all_val = [], [], []

                for t in range(args.num_steps):
                    step_done = b_done[t][mb_idx].to(device)
                    reset = step_done.view(1, -1, 1, 1, 1)
                    mb_h = mb_h * (1.0 - reset)
                    mb_c = mb_c * (1.0 - reset)
                    obs_t  = b_obs[t][mb_idx].to(device)
                    goal_t = b_goal[t][mb_idx].to(device)
                    act_t  = b_act[t][mb_idx].to(device)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                        _, logp, ent, val, mb_h, mb_c = model(
                            obs_t, goal_t, mb_h, mb_c, action=act_t,
                        )
                    all_logp.append(logp)
                    all_ent.append(ent)
                    all_val.append(val)

                logp_seq = torch.stack(all_logp)
                ent_seq  = torch.stack(all_ent)
                val_seq  = torch.stack(all_val)

                old_logp = b_logp[:, mb_idx].to(device)
                adv      = b_adv[:, mb_idx].to(device)
                ret      = b_ret[:, mb_idx].to(device)
                old_val  = b_val[:, mb_idx].to(device)
                ratio    = (logp_seq - old_logp).exp()

                adv_flat = adv.reshape(-1)
                adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
                adv = adv_flat.reshape_as(adv)

                pg1 = -adv * ratio
                pg2 = -adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                v_clip = old_val + (val_seq - old_val).clamp(-args.clip_coef, args.clip_coef)
                v_loss = 0.5 * torch.max(
                    (val_seq - ret).square(), (v_clip - ret).square()
                ).mean()
                ent_loss = ent_seq.mean()
                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                clip_fracs.append(
                    ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                )

        # ── logging ──────────────────────────────────────────────
        sps    = int(global_step / (time.time() - t0))
        y_pred = b_val.numpy().ravel()
        y_true = b_ret.numpy().ravel()
        var_y  = np.var(y_true)
        ev = (1 - np.var(y_true - y_pred) / (var_y + 1e-8)) if var_y > 0 else float("nan")

        wandb.log({
            "losses/policy":        pg_loss.item(),
            "losses/value":         v_loss.item(),
            "losses/entropy":       ent_loss.item(),
            "losses/clip_frac":     np.mean(clip_fracs),
            "losses/explained_var": ev,
            "charts/lr":            lr_now,
            "charts/sps":           sps,
        }, step=global_step)

        if update % 50 == 0:
            print(
                f"[{update}/{num_updates}]  step={global_step:>9,}  sps={sps:>6,}  "
                f"lr={lr_now:.2e}  pg={pg_loss.item():.4f}  "
                f"vl={v_loss.item():.4f}  ent={ent_loss.item():.4f}  ev={ev:.3f}"
            )

        if update % 200 == 0:
            ckpt_path = os.path.join(args.save_dir, f"v3_step{global_step}.pt")
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)

    final = os.path.join(args.save_dir, "v3_final.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, final)
    print(f"saved → {final}")
    wandb.finish()


if __name__ == "__main__":
    main()
