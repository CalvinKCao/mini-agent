"""V1 — CleanRL-style MAPPO with CNN actor-critic on Forked Corridor.

Both agents share a single ActorCritic network (parameter sharing).
Observations are (3,7,7) spatial grids + 2-dim one-hot goal vector.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import wandb
from compile_safe import maybe_compile
from env import ForkedCorridorVec

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
    p.add_argument("--wandb-project",   type=str,   default="forked-corridor-tom")
    p.add_argument("--run-name",        type=str,   default=None)
    p.add_argument("--device",          type=str,   default="cuda")
    p.add_argument("--compile",         action="store_true", default=True)
    p.add_argument("--no-compile",      dest="compile", action="store_false")
    p.add_argument("--save-dir",        type=str,   default="checkpoints")
    return p.parse_args()


# ── network ───────────────────────────────────────────────────────────
def ortho_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            ortho_init(nn.Conv2d(3, 32, 3, padding=1)),
            nn.ReLU(),
            ortho_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.ReLU(),
        )
        # 64*7*7 = 3136 spatial features + 2 goal dims
        self.trunk = nn.Sequential(
            ortho_init(nn.Linear(3136 + 2, 256)),
            nn.ReLU(),
        )
        self.pi_head  = ortho_init(nn.Linear(256, 5), std=0.01)
        self.val_head = ortho_init(nn.Linear(256, 1), std=1.0)

    def _features(self, obs, goal):
        x = self.cnn(obs).flatten(1)           # (B, 3136)
        return self.trunk(torch.cat([x, goal], 1))

    def get_value(self, obs, goal):
        return self.val_head(self._features(obs, goal)).squeeze(-1)

    def get_action_value(self, obs, goal, action=None):
        feat = self._features(obs, goal)
        logits = self.pi_head(feat)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.val_head(feat).squeeze(-1)


# ── training ──────────────────────────────────────────────────────────
def main():
    args = parse_args()
    run_name = args.run_name or f"v1-cnn-ppo-s{args.seed}"

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
            "architecture": "CNN(3→32→64)+MLP(3138→256) ActorCritic, param-shared MAPPO",
        },
    )

    env = ForkedCorridorVec(
        num_envs=args.num_envs, max_steps=args.max_steps, seed=args.seed,
    )

    model = ActorCritic().to(device)
    model, compile_status = maybe_compile(
        model, device=device, enabled=args.compile, mode="reduce-overhead",
    )
    if compile_status != "on_reduce-overhead":
        print(f"train_v1: torch.compile → {compile_status} (eager mode)")
    wandb.config.update({"torch_compile_status": compile_status}, allow_val_change=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    num_updates    = args.total_timesteps // (args.num_envs * args.num_steps)
    batch_size     = args.num_envs * args.num_steps * 2      # ×2 agents
    minibatch_size = batch_size // args.num_minibatches

    # rollout buffers (CPU)
    buf_obs  = torch.zeros(args.num_steps, args.num_envs, 2, 3, 7, 7)
    buf_goal = torch.zeros(args.num_steps, args.num_envs, 2, 2)
    buf_act  = torch.zeros(args.num_steps, args.num_envs, 2, dtype=torch.long)
    buf_logp = torch.zeros(args.num_steps, args.num_envs, 2)
    buf_rew  = torch.zeros(args.num_steps, args.num_envs, 2)
    buf_done = torch.zeros(args.num_steps, args.num_envs)
    buf_val  = torch.zeros(args.num_steps, args.num_envs, 2)

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
                    act, logp, _, val = model.get_action_value(flat_obs, flat_goal)

            act  = act.cpu().reshape(args.num_envs, 2)
            logp = logp.cpu().reshape(args.num_envs, 2)
            val  = val.cpu().reshape(args.num_envs, 2)

            buf_act[step]  = act
            buf_logp[step] = logp
            buf_val[step]  = val

            obs_np, goal_np, rew_np, done_np, info = env.step(act.numpy())

            buf_rew[step] = torch.from_numpy(rew_np)
            next_obs  = torch.from_numpy(obs_np)
            next_goal = torch.from_numpy(goal_np)
            next_done = torch.from_numpy(done_np.astype(np.float32))

            if "episode_returns" in info:
                er = info["episode_returns"]
                el = info["episode_lengths"]
                sr = info["successes"]
                cr = info["collisions"]
                wandb.log({
                    "charts/ep_return_mean":  er.mean(),
                    "charts/ep_return_agent": er.mean(axis=0).tolist(),
                    "charts/ep_length":       el.mean(),
                    "charts/success_rate":    sr.mean(),
                    "charts/collision_rate":  cr.mean(),
                    "charts/global_step":     global_step,
                }, step=global_step)

        # ── GAE ──────────────────────────────────────────────────
        with torch.no_grad():
            nv_obs  = next_obs.reshape(-1, 3, 7, 7).to(device)
            nv_goal = next_goal.reshape(-1, 2).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                next_val = model.get_value(nv_obs, nv_goal)
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

        # flatten: (steps, envs, 2, ...) → (steps*envs*2, ...)
        b_obs  = buf_obs.reshape(-1, 3, 7, 7)
        b_goal = buf_goal.reshape(-1, 2)
        b_act  = buf_act.reshape(-1)
        b_logp = buf_logp.reshape(-1)
        b_adv  = advantages.reshape(-1)
        b_ret  = returns.reshape(-1)
        b_val  = buf_val.reshape(-1)

        # ── PPO update ───────────────────────────────────────────
        inds = np.arange(batch_size)
        clip_fracs = []

        for _epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb = inds[start : start + minibatch_size]

                mb_obs  = b_obs[mb].to(device)
                mb_goal = b_goal[mb].to(device)
                mb_act  = b_act[mb].to(device)
                mb_adv  = b_adv[mb].to(device)
                mb_ret  = b_ret[mb].to(device)
                mb_old_logp = b_logp[mb].to(device)
                mb_old_val  = b_val[mb].to(device)

                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    _, new_logp, ent, new_val = model.get_action_value(
                        mb_obs, mb_goal, mb_act,
                    )

                ratio = (new_logp - mb_old_logp).exp()

                # per-minibatch advantage normalization
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                v_unclipped = (new_val - mb_ret).square()
                v_clipped   = mb_old_val + (new_val - mb_old_val).clamp(
                    -args.clip_coef, args.clip_coef,
                )
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - mb_ret).square()).mean()

                ent_loss = ent.mean()

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
        y_pred, y_true = b_val.numpy(), b_ret.numpy()
        var_y = np.var(y_true)
        ev = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else float("nan")

        wandb.log({
            "losses/policy":       pg_loss.item(),
            "losses/value":        v_loss.item(),
            "losses/entropy":      ent_loss.item(),
            "losses/clip_frac":    np.mean(clip_fracs),
            "losses/explained_var": ev,
            "charts/lr":           lr_now,
            "charts/sps":          sps,
        }, step=global_step)

        if update % 50 == 0:
            print(
                f"[{update}/{num_updates}]  step={global_step:>9,}  "
                f"sps={sps:>6,}  lr={lr_now:.2e}  "
                f"pg={pg_loss.item():.4f}  vl={v_loss.item():.4f}  "
                f"ent={ent_loss.item():.4f}  ev={ev:.3f}"
            )

        if update % 200 == 0:
            ckpt = os.path.join(args.save_dir, f"v1_step{global_step}.pt")
            torch.save(model.state_dict(), ckpt)

    # final save
    final_path = os.path.join(args.save_dir, "v1_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"saved final model → {final_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
