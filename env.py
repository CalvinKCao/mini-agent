"""Forked Corridor — vectorized two-agent coordination environment.

7x7 grid with a shared corridor, two bottleneck passages, and two goal zones.
Agents have hidden private goals (L or R) and must coordinate to avoid
colliding in the bottlenecks while reaching their assigned goals.

Grid layout:
    ####### row 0
    #L...R# row 1  goals: L=(1,1), R=(1,5)
    ##.#.## row 2  bottlenecks: (2,2) and (2,4)
    #.....# row 3  shared corridor
    #.....# row 4  shared corridor / spawn
    #.....# row 5  shared corridor / spawn
    ####### row 6

Movement rules:
 - Agents never share a cell. If both target the same cell or swap,
   both are blocked (stay in place).
 - Bottleneck collision: both targeting the same bottleneck → -5 each,
   episode ends.
 - An agent that has reached its goal is "removed" from physics
   (doesn't block the partner, invisible in obs).
"""

import numpy as np

WALLS = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
], dtype=np.float32)

GOALS = np.array([[1, 1], [1, 5]], dtype=np.int32)        # 0=Left, 1=Right
BOTTLENECKS = np.array([[2, 2], [2, 4]], dtype=np.int32)
SPAWNS = np.array([
    [4, 2], [4, 3], [4, 4],
    [5, 2], [5, 3], [5, 4],
], dtype=np.int32)

# movement deltas: up, down, left, right, stay
DR = np.array([-1, 1, 0, 0, 0], dtype=np.int32)
DC = np.array([0, 0, -1, 1, 0], dtype=np.int32)
N_ACTIONS = 5


class ForkedCorridorVec:
    """Batch of N forked-corridor coordination environments.

    Follows a vectorized ParallelEnv-style API: both agents act
    simultaneously, all N envs step in lockstep.
    """

    def __init__(self, num_envs=512, max_steps=30, seed=None):
        self.n = num_envs
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.pos = np.zeros((num_envs, 2, 2), dtype=np.int32)
        self.goals = np.zeros((num_envs, 2), dtype=np.int64)
        self.reached = np.zeros((num_envs, 2), dtype=bool)
        self.t = np.zeros(num_envs, dtype=np.int32)

        self.ep_ret = np.zeros((num_envs, 2), dtype=np.float64)
        self.ep_len = np.zeros(num_envs, dtype=np.int32)

    # ------------------------------------------------------------------
    def reset(self, mask=None):
        if mask is None:
            mask = np.ones(self.n, dtype=bool)
        idx = np.where(mask)[0]
        k = len(idx)
        if k == 0:
            return self._obs()

        self.goals[idx] = self.rng.integers(0, 2, size=(k, 2))

        noise = self.rng.random((k, len(SPAWNS)))
        picks = noise.argsort(axis=1)[:, :2]
        self.pos[idx, 0] = SPAWNS[picks[:, 0]]
        self.pos[idx, 1] = SPAWNS[picks[:, 1]]

        self.reached[idx] = False
        self.t[idx] = 0
        self.ep_ret[idx] = 0.0
        self.ep_len[idx] = 0

        return self._obs()

    # ------------------------------------------------------------------
    def _obs(self):
        """Per-agent observations.

        Returns
        -------
        obs      : (N, 2, 3, 7, 7) — channels: walls, self pos, partner pos
        goal_vec : (N, 2, 2)        — one-hot own-goal

        Partner channel is zeroed when the partner has already reached
        its goal (removed from the grid).
        """
        obs = np.zeros((self.n, 2, 3, 7, 7), dtype=np.float32)
        ar = np.arange(self.n)

        obs[:, :, 0] = WALLS

        for a in range(2):
            r, c = self.pos[:, a, 0], self.pos[:, a, 1]
            # self channel — always visible
            obs[ar, a, 1, r, c] = 1.0
            # partner channel — only if partner is still active
            p = 1 - a
            alive = ~self.reached[:, p]
            if alive.any():
                ai = ar[alive]
                pr, pc = self.pos[alive, p, 0], self.pos[alive, p, 1]
                obs[ai, a, 2, pr, pc] = 1.0

        gv = np.zeros((self.n, 2, 2), dtype=np.float32)
        for a in range(2):
            gv[ar, a, self.goals[:, a]] = 1.0

        return obs, gv

    # ------------------------------------------------------------------
    def step(self, actions):
        """Step all envs.

        Parameters
        ----------
        actions : (N, 2) int — one action per agent per env

        Returns
        -------
        obs, goal_vec, rewards (N,2), dones (N,), infos dict
        """
        rewards = np.zeros((self.n, 2), dtype=np.float32)

        for a in range(2):
            rewards[~self.reached[:, a], a] -= 0.1

        # ── desired positions (wall + freeze check) ──────────────
        desired = self.pos.copy()
        for a in range(2):
            act = actions[:, a]
            nr = np.clip(self.pos[:, a, 0] + DR[act], 0, 6)
            nc = np.clip(self.pos[:, a, 1] + DC[act], 0, 6)
            hit = WALLS[nr, nc] > 0.5
            nr = np.where(hit, self.pos[:, a, 0], nr)
            nc = np.where(hit, self.pos[:, a, 1], nc)
            active = ~self.reached[:, a]
            desired[:, a, 0] = np.where(active, nr, self.pos[:, a, 0])
            desired[:, a, 1] = np.where(active, nc, self.pos[:, a, 1])

        # ── conflict resolution (active agents only) ─────────────
        d0r, d0c = desired[:, 0, 0], desired[:, 0, 1]
        d1r, d1c = desired[:, 1, 0], desired[:, 1, 1]
        c0r, c0c = self.pos[:, 0, 0], self.pos[:, 0, 1]
        c1r, c1c = self.pos[:, 1, 0], self.pos[:, 1, 1]

        both_active = ~self.reached[:, 0] & ~self.reached[:, 1]

        same_tgt = (d0r == d1r) & (d0c == d1c) & both_active
        swap = ((d0r == c1r) & (d0c == c1c) &
                (d1r == c0r) & (d1c == c0c) &
                ((d0r != c0r) | (d0c != c0c)) &
                ((d1r != c1r) | (d1c != c1c)) &
                both_active)

        blocked = same_tgt | swap

        self.pos[:, 0, 0] = np.where(blocked, c0r, d0r)
        self.pos[:, 0, 1] = np.where(blocked, c0c, d0c)
        self.pos[:, 1, 0] = np.where(blocked, c1r, d1r)
        self.pos[:, 1, 1] = np.where(blocked, c1c, d1c)

        # ── bottleneck collision ─────────────────────────────────
        collision = np.zeros(self.n, dtype=bool)
        for bn_r, bn_c in BOTTLENECKS:
            collision |= same_tgt & (d0r == bn_r) & (d0c == bn_c)

        rewards[collision, 0] -= 5.0
        rewards[collision, 1] -= 5.0

        # ── goal reaching ────────────────────────────────────────
        for a in range(2):
            gr = GOALS[self.goals[:, a], 0]
            gc = GOALS[self.goals[:, a], 1]
            at_goal = (self.pos[:, a, 0] == gr) & (self.pos[:, a, 1] == gc)
            just = at_goal & ~self.reached[:, a] & ~collision
            rewards[just, a] += 10.0
            self.reached[just, a] = True

        self.t += 1
        both_done = self.reached[:, 0] & self.reached[:, 1]
        timeout = self.t >= self.max_steps
        dones = collision | both_done | timeout

        self.ep_ret += rewards
        self.ep_len += 1

        infos = {}
        if dones.any():
            di = np.where(dones)[0]
            infos["episode_returns"] = self.ep_ret[di].copy()
            infos["episode_lengths"] = self.ep_len[di].copy()
            infos["collisions"] = collision[di].copy()
            infos["successes"] = both_done[di].copy()
            self.reset(dones)

        obs, gv = self._obs()
        return obs, gv, rewards, dones, infos


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = ForkedCorridorVec(num_envs=8, max_steps=15, seed=42)
    obs, gv = env.reset()
    assert obs.shape == (8, 2, 3, 7, 7)
    assert gv.shape == (8, 2, 2)

    for _ in range(15):
        acts = env.rng.integers(0, N_ACTIONS, size=(8, 2))
        obs, gv, rew, done, info = env.step(acts)
        # verify no cell sharing between active agents
        for i in range(8):
            if not env.reached[i, 0] and not env.reached[i, 1]:
                assert not (env.pos[i, 0, 0] == env.pos[i, 1, 0] and
                            env.pos[i, 0, 1] == env.pos[i, 1, 1]), \
                    f"cell sharing in env {i}"

    print(f"obs shape: {obs.shape}, goal shape: {gv.shape}")
    print("env smoke test passed (no cell sharing)")
