"""V2 — DRC(3,3) MAPPO on Forked Corridor.

Differences from V1:
  - Recurrent DRC actor-critic replaces stateless CNN
  - Hidden states persist across timesteps, reset on episode boundaries
  - PPO update uses sequence-level mini-batching with BPTT
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import wandb
from env import ForkedCorridorVec
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
    # DRC architecture
    p.add_argument("--drc-depth",       type=int,   default=3,  help="D: ConvLSTM layers")
    p.add_argument("--drc-ticks",       type=int,   default=3,  help="N: internal ticks")
    p.add_argument("--drc-channels",    type=int,   default=32, help="G: hidden channels")
    # infra
    p.add_argument("--wandb-project",   type=str,   default="forked-corridor-tom")
    p.add_argument("--run-name",        type=str,   default=None)
    p.add_argument("--device",          type=str,   default="cuda")
    p.add_argument("--compile",         action="store_true", default=True)
    p.add_argument("--no-compile",      dest="compile", action="store_false")
    p.add_argument("--save-dir",        type=str,   default="checkpoints")
    return p.parse_args()


# ── training ──────────────────────────────────────────────────────────
def main():
    args = parse_args()
    run_name = args.run_name or f"v2-drc{args.drc_depth}{args.drc_ticks}-s{args.seed}"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            **vars(args),
            "architecture": f"DRC({args.drc_depth},{args.drc_ticks}) G={args.drc_channels}, "
                            "pool-and-inject, BU+TD skips, param-shared MAPPO",
        },
    )

    env = ForkedCorridorVec(
        num_envs=args.num_envs, max_steps=args.max_steps, seed=args.seed,
    )

    model = DRCActorCritic(
        obs_ch=3, goal_dim=2,
        G=args.drc_channels, D=args.drc_depth, N=args.drc_ticks,
    ).to(device)
    if args.compile and device.type == "cuda":
        model = torch.compile(model, mode="default")

    # weight decay only on policy/value heads (per paper E.4)
    head_params = set()
    for n, p in model.named_parameters():
        if "pi_head" in n or "val_head" in n:
            head_params.add(id(p))
    param_groups = [
        {"params": [p for p in model.parameters() if id(p) not in head_params],
         "weight_decay": 0.0},
        {"params": [p for p in model.parameters() if id(p) in head_params],
         "weight_decay": args.weight_decay},
    ]
    optimizer = torch.optim.Adam(param_groups, lr=args.lr)

    n_agents       = args.num_envs * 2
    num_updates    = args.total_timesteps // (args.num_envs * args.num_steps)
    seq_batch_size = n_agents                                 # total sequences
    mb_seq_count   = seq_batch_size // args.num_minibatches   # seqs per minibatch

    # rollout buffers (CPU, agent-flattened where needed)
    buf_obs  = torch.zeros(args.num_steps, args.num_envs, 2, 3, 7, 7)
    buf_goal = torch.zeros(args.num_steps, args.num_envs, 2, 2)
    buf_act  = torch.zeros(args.num_steps, args.num_envs, 2, dtype=torch.long)
    buf_logp = torch.zeros(args.num_steps, args.num_envs, 2)
    buf_rew  = torch.zeros(args.num_steps, args.num_envs, 2)
    buf_done = torch.zeros(args.num_steps, args.num_envs)
    buf_val  = torch.zeros(args.num_steps, args.num_envs, 2)

    # hidden states live on GPU; shape (D, n_agents, G, 7, 7)
    h, c = model.initial_state(n_agents, device=device)

    obs_np, goal_np = env.reset()
    next_obs  = torch.from_numpy(obs_np)
    next_goal = torch.from_numpy(goal_np)
    next_done = torch.zeros(args.num_envs)

    global_step = 0
    t0 = time.time()
    os.makedirs(args.save_dir, exist_ok=True)

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1) / num_updates
        lr_now = frac * args.lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # snapshot hidden states for the PPO re-run
        init_h = h.detach().clone()
        init_c = c.detach().clone()

        # ── rollout ──────────────────────────────────────────────
        for step in range(args.num_steps):
            global_step += args.num_envs

            buf_obs[step]  = next_obs
            buf_goal[step] = next_goal
            buf_done[step] = next_done

            flat_obs  = next_obs.reshape(-1, 3, 7, 7).to(device)
            flat_goal = next_goal.reshape(-1, 2).to(device)

            with torch.no_grad():
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    act, logp, _, val, h, c = model(flat_obs, flat_goal, h, c)

            act_cpu  = act.cpu().reshape(args.num_envs, 2)
            logp_cpu = logp.cpu().reshape(args.num_envs, 2)
            val_cpu  = val.cpu().reshape(args.num_envs, 2)

            buf_act[step]  = act_cpu
            buf_logp[step] = logp_cpu
            buf_val[step]  = val_cpu

            obs_np, goal_np, rew_np, done_np, info = env.step(act_cpu.numpy())

            buf_rew[step] = torch.from_numpy(rew_np)
            next_obs  = torch.from_numpy(obs_np)
            next_goal = torch.from_numpy(goal_np)
            next_done = torch.from_numpy(done_np.astype(np.float32))

            # reset hidden states for finished envs
            if done_np.any():
                # expand env-level done → agent-level
                dm = torch.from_numpy(done_np).to(device)
                agent_reset = dm.unsqueeze(-1).expand(-1, 2).reshape(-1).bool()
                h[:, agent_reset] = 0
                c[:, agent_reset] = 0

            if "episode_returns" in info:
                wandb.log({
                    "charts/ep_return_mean": info["episode_returns"].mean(),
                    "charts/ep_length":      info["episode_lengths"].mean(),
                    "charts/success_rate":   info["successes"].mean(),
                    "charts/collision_rate": info["collisions"].mean(),
                    "charts/global_step":    global_step,
                }, step=global_step)

        # ── GAE (same as V1, per-agent) ──────────────────────────
        with torch.no_grad():
            nv_obs  = next_obs.reshape(-1, 3, 7, 7).to(device)
            nv_goal = next_goal.reshape(-1, 2).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                next_val, h, c = model.get_value(nv_obs, nv_goal, h, c)
            next_val = next_val.cpu().reshape(args.num_envs, 2)

        advantages = torch.zeros_like(buf_rew)
        lastgae = torch.zeros(args.num_envs, 2)

        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nnt = (1.0 - next_done).unsqueeze(-1)
                nv  = next_val
            else:
                nnt = (1.0 - buf_done[t + 1]).unsqueeze(-1)
                nv  = buf_val[t + 1]
            delta = buf_rew[t] + args.gamma * nv * nnt - buf_val[t]
            lastgae = delta + args.gamma * args.gae_lambda * nnt * lastgae
            advantages[t] = lastgae

        returns = advantages + buf_val

        # reshape to agent-level: (steps, n_agents, ...)
        b_obs  = buf_obs.reshape(args.num_steps, n_agents, 3, 7, 7)
        b_goal = buf_goal.reshape(args.num_steps, n_agents, 2)
        b_act  = buf_act.reshape(args.num_steps, n_agents)
        b_logp = buf_logp.reshape(args.num_steps, n_agents)
        b_adv  = advantages.reshape(args.num_steps, n_agents)
        b_ret  = returns.reshape(args.num_steps, n_agents)
        b_val  = buf_val.reshape(args.num_steps, n_agents)
        # env-level done → agent-level
        b_done = buf_done.unsqueeze(-1).expand(-1, -1, 2).reshape(
            args.num_steps, n_agents)

        # ── PPO update (sequence mini-batches) ───────────────────
        clip_fracs = []

        for _epoch in range(args.update_epochs):
            perm = np.random.permutation(n_agents)

            for mb_start in range(0, n_agents, mb_seq_count):
                mb_idx = perm[mb_start: mb_start + mb_seq_count]

                # initial hidden state for this mini-batch
                mb_h = init_h[:, mb_idx].clone()
                mb_c = init_c[:, mb_idx].clone()

                all_logp, all_ent, all_val = [], [], []

                for t in range(args.num_steps):
                    # mask-based hidden state reset (no in-place on graph)
                    step_done = b_done[t][mb_idx].to(device)      # (mb,)
                    reset = step_done.view(1, -1, 1, 1, 1)        # broadcast
                    mb_h = mb_h * (1.0 - reset)
                    mb_c = mb_c * (1.0 - reset)

                    obs_t  = b_obs[t][mb_idx].to(device)
                    goal_t = b_goal[t][mb_idx].to(device)
                    act_t  = b_act[t][mb_idx].to(device)

                    with torch.autocast(device.type, dtype=torch.bfloat16,
                                        enabled=use_amp):
                        _, logp, ent, val, mb_h, mb_c = model(
                            obs_t, goal_t, mb_h, mb_c, action=act_t,
                        )

                    all_logp.append(logp)
                    all_ent.append(ent)
                    all_val.append(val)

                logp_seq = torch.stack(all_logp)   # (steps, mb)
                ent_seq  = torch.stack(all_ent)
                val_seq  = torch.stack(all_val)

                old_logp = b_logp[:, mb_idx].to(device)
                adv      = b_adv[:, mb_idx].to(device)
                ret      = b_ret[:, mb_idx].to(device)
                old_val  = b_val[:, mb_idx].to(device)

                ratio = (logp_seq - old_logp).exp()

                adv_flat = adv.reshape(-1)
                adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
                adv = adv_flat.reshape_as(adv)

                pg1 = -adv * ratio
                pg2 = -adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                v_unclip = (val_seq - ret).square()
                v_clip   = old_val + (val_seq - old_val).clamp(
                    -args.clip_coef, args.clip_coef)
                v_loss = 0.5 * torch.max(v_unclip, (v_clip - ret).square()).mean()

                ent_loss = ent_seq.mean()

                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    clip_fracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

        # ── logging ──────────────────────────────────────────────
        sps = int(global_step / (time.time() - t0))
        y_pred = b_val.numpy().ravel()
        y_true = b_ret.numpy().ravel()
        var_y  = np.var(y_true)
        ev = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else float("nan")

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
                f"[{update}/{num_updates}]  step={global_step:>9,}  "
                f"sps={sps:>6,}  lr={lr_now:.2e}  "
                f"pg={pg_loss.item():.4f}  vl={v_loss.item():.4f}  "
                f"ent={ent_loss.item():.4f}  ev={ev:.3f}"
            )

        if update % 200 == 0:
            ckpt = os.path.join(args.save_dir, f"v2_step{global_step}.pt")
            state = {"model": model.state_dict(), "args": vars(args)}
            torch.save(state, ckpt)

    # final checkpoint
    final = os.path.join(args.save_dir, "v2_final.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, final)
    print(f"saved final model → {final}")
    wandb.finish()


if __name__ == "__main__":
    main()
