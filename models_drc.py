"""DRC(D,N) actor-critic — Deep Repeated ConvLSTM (Guez et al. 2019).

Matches the DRC(3,3) spec from Bush et al. (2025):
  - 3 stacked ConvLSTM layers, 3 internal ticks per timestep
  - 32-channel hidden/cell states, 3×3 kernels, padding=1
  - Pool-and-inject, bottom-up skips, top-down skips
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _ortho(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell with pool-and-inject."""

    def __init__(self, input_ch, hidden_ch, kernel_size=3, H=7, W=7):
        super().__init__()
        self.G = hidden_ch
        self.H, self.W = H, W

        # pool-and-inject: global {mean,max} pool → linear → reshape to spatial
        self.pool_proj = _ortho(nn.Linear(2 * hidden_ch, H * W * hidden_ch))

        # gate conv: cat([x, h+p]) → 4*G  (i, f, o, g)
        self.gates = nn.Conv2d(
            input_ch + hidden_ch, 4 * hidden_ch,
            kernel_size, padding=kernel_size // 2,
        )
        nn.init.orthogonal_(self.gates.weight, np.sqrt(2))
        nn.init.zeros_(self.gates.bias)
        self.gates.bias.data[hidden_ch: 2 * hidden_ch] = 1.0  # forget-gate → 1

    def forward(self, x, h, c):
        # pool-and-inject on prior-tick hidden state
        h_mean = h.mean(dim=[2, 3])                       # (B, G)
        h_max  = h.amax(dim=[2, 3])                        # (B, G)
        p = self.pool_proj(torch.cat([h_mean, h_max], 1))  # (B, H*W*G)
        p = p.view(-1, self.G, self.H, self.W)

        combined = torch.cat([x, h + p], dim=1)
        raw = self.gates(combined)
        i, f, o, g = raw.chunk(4, dim=1)

        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new


class DRCActorCritic(nn.Module):
    """DRC(D,N) actor-critic with parameter-shared MAPPO heads.

    Hidden state shape: (D, B, G, H, W) for both h and c,
    where B is the flattened agent dimension (num_envs × 2).
    """

    def __init__(self, obs_ch=3, goal_dim=2, G=32, D=3, N=3, H=7, W=7):
        super().__init__()
        self.D, self.N, self.G = D, N, G
        self.H, self.W = H, W

        self.encoder = _ortho(nn.Conv2d(obs_ch, G, 3, padding=1))

        # D ConvLSTM cells — each receives 2*G input channels
        # (encoder + either top-down skip or inter-layer)
        self.cells = nn.ModuleList([
            ConvLSTMCell(2 * G, G, kernel_size=3, H=H, W=W)
            for _ in range(D)
        ])

        # output: flatten(cat(h_top, i_t)) + goal → trunk → heads
        self.trunk = nn.Sequential(
            _ortho(nn.Linear(2 * G * H * W + goal_dim, 256)),
            nn.ReLU(),
        )
        self.pi_head  = _ortho(nn.Linear(256, 5), std=0.01)
        self.val_head = _ortho(nn.Linear(256, 1), std=1.0)

    # ----- state helpers --------------------------------------------------
    def initial_state(self, batch_size, device=None):
        shape = (self.D, batch_size, self.G, self.H, self.W)
        return (torch.zeros(shape, device=device),
                torch.zeros(shape, device=device))

    # ----- core -----------------------------------------------------------
    def _core(self, i_t, h, c):
        """Run D layers × N ticks.  Returns (feat, new_h, new_c)."""
        hs = [h[d] for d in range(self.D)]
        cs = [c[d] for d in range(self.D)]

        for _tick in range(self.N):
            h_top = hs[self.D - 1]          # top-down from previous tick
            for d in range(self.D):
                skip = h_top if d == 0 else hs[d - 1]
                x = torch.cat([i_t, skip], dim=1)   # bottom-up + skip
                hs[d], cs[d] = self.cells[d](x, hs[d], cs[d])

        return hs, cs

    def _head_features(self, hs, i_t, goal):
        out = torch.cat([hs[self.D - 1], i_t], dim=1).flatten(1)
        return self.trunk(torch.cat([out, goal], dim=1))

    # ----- public API -----------------------------------------------------
    def forward(self, obs, goal, h, c, action=None):
        """Full forward: encode → DRC ticks → policy + value.

        Returns (action, logprob, entropy, value, new_h, new_c).
        """
        i_t = torch.relu(self.encoder(obs))
        hs, cs = self._core(i_t, h, c)

        feat = self._head_features(hs, i_t, goal)
        logits = self.pi_head(feat)
        value  = self.val_head(feat).squeeze(-1)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        new_h = torch.stack(hs)
        new_c = torch.stack(cs)
        return action, dist.log_prob(action), dist.entropy(), value, new_h, new_c

    def get_value(self, obs, goal, h, c):
        """Value-only pass (skips policy head). Returns (value, new_h, new_c)."""
        i_t = torch.relu(self.encoder(obs))
        hs, cs = self._core(i_t, h, c)
        feat = self._head_features(hs, i_t, goal)
        new_h = torch.stack(hs)
        new_c = torch.stack(cs)
        return self.val_head(feat).squeeze(-1), new_h, new_c
