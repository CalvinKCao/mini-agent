"""Gridworld environments for partner-intent coordination experiments.

ThreeCorridorVec (primary, V3):
  9×7 grid with three narrow bottleneck passages and three goal zones.
  Each agent has a private hidden corridor intent (LEFT=0, CENTER=1, RIGHT=2).
  Agents must infer the partner's intended corridor from early movement cues
  to coordinate routing and avoid costly bottleneck collisions.

  Grid layout (9 cols × 7 rows):
      ######### row 0
      #L..C..R# row 1  goals: L=(1,1), C=(1,4), R=(1,7)
      ##.#.#.## row 2  bottlenecks: (2,2), (2,4), (2,6)
      #.......# row 3  shared corridor
      #.......# row 4  shared corridor / spawn
      #.......# row 5  spawn zone
      ######### row 6

ForkedCorridorVec (legacy, V1/V2):
  Original 7×7 two-corridor environment. Kept for backward compat.
"""

import numpy as np

# ── Three-corridor constants ───────────────────────────────────────────
GRID_H, GRID_W = 7, 9
N_CORRIDORS = 3

WALLS_3C = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=np.float32)

# 0=LEFT, 1=CENTER, 2=RIGHT
GOALS_3C = np.array([[1, 1], [1, 4], [1, 7]], dtype=np.int32)
BOTTLENECK_COLS = np.array([2, 4, 6], dtype=np.int32)
BOTTLENECKS_3C = np.array([[2, 2], [2, 4], [2, 6]], dtype=np.int32)
CORRIDOR_NAMES = ["LEFT", "CENTER", "RIGHT"]

SPAWNS_3C = np.array([
    [4, 2], [4, 3], [4, 4], [4, 5], [4, 6],
    [5, 2], [5, 3], [5, 4], [5, 5], [5, 6],
], dtype=np.int32)

# ── Shared action constants ────────────────────────────────────────────
DR = np.array([-1, 1, 0, 0, 0], dtype=np.int32)   # up, down, left, right, stay
DC = np.array([0, 0, -1, 1, 0], dtype=np.int32)
N_ACTIONS = 5


# ── Scripted partner ──────────────────────────────────────────────────
def scripted_actions_vec(pos, goals, reached):
    """Vectorized deterministic partner: beeline to bottleneck column, then up to goal.

    Parameters
    ----------
    pos     : (N, 2) int — current row, col
    goals   : (N,)   int — corridor index 0/1/2
    reached : (N,)   bool

    Returns
    -------
    (N,) int — action for each env
    """
    N = len(goals)
    actions = np.full(N, 4, dtype=np.int32)
    active = ~reached
    if not active.any():
        return actions

    r = pos[active, 0]
    c = pos[active, 1]
    g = goals[active]
    bc = BOTTLENECK_COLS[g]    # target bottleneck column
    gc = GOALS_3C[g, 1]        # target goal column

    a = np.full(int(active.sum()), 4, dtype=np.int32)

    # rows 3-5: align to bottleneck column, then go up
    in_open = r >= 3
    a = np.where(in_open & (c < bc), 3, a)
    a = np.where(in_open & (c > bc), 2, a)
    a = np.where(in_open & (c == bc), 0, a)

    # row 2 (bottleneck cell): go up
    a = np.where(r == 2, 0, a)

    # row 1 (goal row): move toward goal column
    in1 = r == 1
    a = np.where(in1 & (c < gc), 3, a)
    a = np.where(in1 & (c > gc), 2, a)

    actions[active] = a
    return actions


# ── Three-Corridor environment ────────────────────────────────────────
class ThreeCorridorVec:
    """Vectorized three-corridor coordination environment.

    Both agents share one ActorCritic network (param-sharing MAPPO).
    Each agent receives:
      obs      : (3, 7, 9) — walls / self / partner (masked when mask_partner_obs=True)
      goal_vec : (3,)       — one-hot own corridor intent

    Key attributes (public, readable mid-episode):
      pos                : (N, 2, 2) int  — row/col per env/agent
      goals              : (N, 2)    int  — corridor intent per env/agent
      reached            : (N, 2)    bool
      corridor_committed : (N, 2)    int  — corridor idx first time agent enters row 2 (-1 = not yet)
      t                  : (N,)      int  — step counter
    """

    def __init__(self, num_envs=512, max_steps=30, seed=None, mask_partner_obs=False):
        self.n = num_envs
        self.max_steps = max_steps
        self.mask_partner_obs = mask_partner_obs
        self.rng = np.random.default_rng(seed)

        self.pos                = np.zeros((num_envs, 2, 2), dtype=np.int32)
        self.goals              = np.zeros((num_envs, 2), dtype=np.int64)
        self.reached            = np.zeros((num_envs, 2), dtype=bool)
        self.corridor_committed = np.full((num_envs, 2), -1, dtype=np.int32)
        self.t                  = np.zeros(num_envs, dtype=np.int32)
        self.ep_ret             = np.zeros((num_envs, 2), dtype=np.float64)
        self.ep_len             = np.zeros(num_envs, dtype=np.int32)

    # ------------------------------------------------------------------
    def reset(self, mask=None):
        if mask is None:
            mask = np.ones(self.n, dtype=bool)
        idx = np.where(mask)[0]
        k = len(idx)
        if k == 0:
            return self._obs()

        self.goals[idx] = self.rng.integers(0, N_CORRIDORS, size=(k, 2))

        # pick 2 distinct spawn positions per env
        noise = self.rng.random((k, len(SPAWNS_3C)))
        picks = noise.argsort(axis=1)[:, :2]
        self.pos[idx, 0] = SPAWNS_3C[picks[:, 0]]
        self.pos[idx, 1] = SPAWNS_3C[picks[:, 1]]

        self.reached[idx]            = False
        self.corridor_committed[idx] = -1
        self.t[idx]                  = 0
        self.ep_ret[idx]             = 0.0
        self.ep_len[idx]             = 0

        return self._obs()

    # ------------------------------------------------------------------
    def _obs(self):
        """Build per-agent observations.

        Returns
        -------
        obs      : (N, 2, 3, 7, 9) — walls / self / partner
        goal_vec : (N, 2, 3)        — one-hot own-goal corridor
        """
        obs = np.zeros((self.n, 2, 3, GRID_H, GRID_W), dtype=np.float32)
        ar = np.arange(self.n)

        obs[:, :, 0] = WALLS_3C

        for a in range(2):
            r, c = self.pos[:, a, 0], self.pos[:, a, 1]
            obs[ar, a, 1, r, c] = 1.0

            if not self.mask_partner_obs:
                p = 1 - a
                alive = ~self.reached[:, p]
                if alive.any():
                    ai = ar[alive]
                    pr, pc = self.pos[alive, p, 0], self.pos[alive, p, 1]
                    obs[ai, a, 2, pr, pc] = 1.0

        gv = np.zeros((self.n, 2, N_CORRIDORS), dtype=np.float32)
        for a in range(2):
            gv[ar, a, self.goals[:, a]] = 1.0

        return obs, gv

    # ------------------------------------------------------------------
    def step(self, actions):
        """Step all envs simultaneously.

        Parameters
        ----------
        actions : (N, 2) int

        Returns
        -------
        obs, goal_vec, rewards (N,2), dones (N,), infos dict

        infos keys (only populated when dones.any()):
          episode_returns   : (k, 2)
          episode_lengths   : (k,)
          partner_intent    : (k,)   — B's corridor (goals[:, 1])
          chosen_corridor_a : (k,)   — A's first bottleneck entered (-1 if none)
          chosen_corridor_b : (k,)   — B's first bottleneck entered (-1 if none)
          collisions        : (k,)   bool
          successes         : (k,)   bool
          corridor_match    : (k,)   bool — A chose a different corridor from B
        """
        rewards = np.zeros((self.n, 2), dtype=np.float32)
        for a in range(2):
            rewards[~self.reached[:, a], a] -= 0.1

        # ── desired positions ─────────────────────────────────────────
        desired = self.pos.copy()
        for a in range(2):
            act = actions[:, a]
            nr = np.clip(self.pos[:, a, 0] + DR[act], 0, GRID_H - 1)
            nc = np.clip(self.pos[:, a, 1] + DC[act], 0, GRID_W - 1)
            hit = WALLS_3C[nr, nc] > 0.5
            nr = np.where(hit, self.pos[:, a, 0], nr)
            nc = np.where(hit, self.pos[:, a, 1], nc)
            active = ~self.reached[:, a]
            desired[:, a, 0] = np.where(active, nr, self.pos[:, a, 0])
            desired[:, a, 1] = np.where(active, nc, self.pos[:, a, 1])

        # ── conflict resolution ───────────────────────────────────────
        d0r, d0c = desired[:, 0, 0], desired[:, 0, 1]
        d1r, d1c = desired[:, 1, 0], desired[:, 1, 1]
        c0r, c0c = self.pos[:, 0, 0], self.pos[:, 0, 1]
        c1r, c1c = self.pos[:, 1, 0], self.pos[:, 1, 1]

        both_active = ~self.reached[:, 0] & ~self.reached[:, 1]
        same_tgt = (d0r == d1r) & (d0c == d1c) & both_active
        swap = (
            (d0r == c1r) & (d0c == c1c) &
            (d1r == c0r) & (d1c == c0c) &
            ((d0r != c0r) | (d0c != c0c)) &
            ((d1r != c1r) | (d1c != c1c)) &
            both_active
        )
        blocked = same_tgt | swap

        self.pos[:, 0, 0] = np.where(blocked, c0r, d0r)
        self.pos[:, 0, 1] = np.where(blocked, c0c, d0c)
        self.pos[:, 1, 0] = np.where(blocked, c1r, d1r)
        self.pos[:, 1, 1] = np.where(blocked, c1c, d1c)

        # ── bottleneck collision ──────────────────────────────────────
        collision = np.zeros(self.n, dtype=bool)
        for bn_r, bn_c in BOTTLENECKS_3C:
            collision |= same_tgt & (d0r == bn_r) & (d0c == bn_c)
        rewards[collision, 0] -= 5.0
        rewards[collision, 1] -= 5.0

        # ── corridor commitment (first entry into row 2) ──────────────
        for ci, bc in enumerate(BOTTLENECK_COLS):
            for a in range(2):
                not_yet = self.corridor_committed[:, a] == -1
                at_btn  = (self.pos[:, a, 0] == 2) & (self.pos[:, a, 1] == bc)
                self.corridor_committed[not_yet & at_btn, a] = ci

        # ── goal reaching ─────────────────────────────────────────────
        for a in range(2):
            gr = GOALS_3C[self.goals[:, a], 0]
            gc = GOALS_3C[self.goals[:, a], 1]
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
            infos["episode_returns"]   = self.ep_ret[di].copy()
            infos["episode_lengths"]   = self.ep_len[di].copy()
            infos["partner_intent"]    = self.goals[di, 1].copy()
            infos["chosen_corridor_a"] = self.corridor_committed[di, 0].copy()
            infos["chosen_corridor_b"] = self.corridor_committed[di, 1].copy()
            infos["collisions"]        = collision[di].copy()
            infos["successes"]         = both_done[di].copy()
            ca = self.corridor_committed[di, 0]
            cb = self.corridor_committed[di, 1]
            infos["corridor_match"]    = (ca != -1) & (cb != -1) & (ca != cb)
            self.reset(dones)

        obs, gv = self._obs()
        return obs, gv, rewards, dones, infos


# ── Legacy two-corridor environment (V1/V2) ───────────────────────────
WALLS = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
], dtype=np.float32)

GOALS       = np.array([[1, 1], [1, 5]], dtype=np.int32)
BOTTLENECKS = np.array([[2, 2], [2, 4]], dtype=np.int32)
SPAWNS      = np.array([[4,2],[4,3],[4,4],[5,2],[5,3],[5,4]], dtype=np.int32)


class ForkedCorridorVec:
    """Legacy 7×7 two-corridor environment (backward compat for V1/V2)."""

    def __init__(self, num_envs=512, max_steps=30, seed=None):
        self.n = num_envs
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.pos     = np.zeros((num_envs, 2, 2), dtype=np.int32)
        self.goals   = np.zeros((num_envs, 2), dtype=np.int64)
        self.reached = np.zeros((num_envs, 2), dtype=bool)
        self.t       = np.zeros(num_envs, dtype=np.int32)
        self.ep_ret  = np.zeros((num_envs, 2), dtype=np.float64)
        self.ep_len  = np.zeros(num_envs, dtype=np.int32)

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

    def _obs(self):
        obs = np.zeros((self.n, 2, 3, 7, 7), dtype=np.float32)
        ar  = np.arange(self.n)
        obs[:, :, 0] = WALLS
        for a in range(2):
            r, c = self.pos[:, a, 0], self.pos[:, a, 1]
            obs[ar, a, 1, r, c] = 1.0
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

    def step(self, actions):
        rewards = np.zeros((self.n, 2), dtype=np.float32)
        for a in range(2):
            rewards[~self.reached[:, a], a] -= 0.1
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
        d0r, d0c = desired[:, 0, 0], desired[:, 0, 1]
        d1r, d1c = desired[:, 1, 0], desired[:, 1, 1]
        c0r, c0c = self.pos[:, 0, 0], self.pos[:, 0, 1]
        c1r, c1c = self.pos[:, 1, 0], self.pos[:, 1, 1]
        both_active = ~self.reached[:, 0] & ~self.reached[:, 1]
        same_tgt = (d0r == d1r) & (d0c == d1c) & both_active
        swap = ((d0r == c1r) & (d0c == c1c) & (d1r == c0r) & (d1c == c0c) &
                ((d0r != c0r) | (d0c != c0c)) & ((d1r != c1r) | (d1c != c1c)) & both_active)
        blocked = same_tgt | swap
        self.pos[:, 0, 0] = np.where(blocked, c0r, d0r)
        self.pos[:, 0, 1] = np.where(blocked, c0c, d0c)
        self.pos[:, 1, 0] = np.where(blocked, c1r, d1r)
        self.pos[:, 1, 1] = np.where(blocked, c1c, d1c)
        collision = np.zeros(self.n, dtype=bool)
        for bn_r, bn_c in BOTTLENECKS:
            collision |= same_tgt & (d0r == bn_r) & (d0c == bn_c)
        rewards[collision, 0] -= 5.0
        rewards[collision, 1] -= 5.0
        for a in range(2):
            gr = GOALS[self.goals[:, a], 0]
            gc = GOALS[self.goals[:, a], 1]
            at_goal = (self.pos[:, a, 0] == gr) & (self.pos[:, a, 1] == gc)
            just = at_goal & ~self.reached[:, a] & ~collision
            rewards[just, a] += 10.0
            self.reached[just, a] = True
        self.t += 1
        both_done = self.reached[:, 0] & self.reached[:, 1]
        dones = collision | both_done | (self.t >= self.max_steps)
        self.ep_ret += rewards
        self.ep_len += 1
        infos = {}
        if dones.any():
            di = np.where(dones)[0]
            infos["episode_returns"] = self.ep_ret[di].copy()
            infos["episode_lengths"] = self.ep_len[di].copy()
            infos["collisions"]      = collision[di].copy()
            infos["successes"]       = (self.reached[:, 0] & self.reached[:, 1])[di].copy()
            self.reset(dones)
        obs, gv = self._obs()
        return obs, gv, rewards, dones, infos


# ── Smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = ThreeCorridorVec(num_envs=8, max_steps=20, seed=42)
    obs, gv = env.reset()
    assert obs.shape == (8, 2, 3, GRID_H, GRID_W), obs.shape
    assert gv.shape == (8, 2, N_CORRIDORS), gv.shape

    for _ in range(20):
        acts = env.rng.integers(0, N_ACTIONS, size=(8, 2))
        obs, gv, rew, done, info = env.step(acts)
        for i in range(8):
            if not env.reached[i, 0] and not env.reached[i, 1]:
                assert not (
                    env.pos[i, 0, 0] == env.pos[i, 1, 0] and
                    env.pos[i, 0, 1] == env.pos[i, 1, 1]
                ), f"cell sharing in env {i}"

    # test scripted partner
    pos_test = np.array([[5, 3], [4, 6]], dtype=np.int32)
    goals_test = np.array([1, 2], dtype=np.int64)
    reached_test = np.array([False, False])
    sa = scripted_actions_vec(pos_test, goals_test, reached_test)
    # agent 0 at (5,3) with goal CENTER (bc=4): should go right
    assert sa[0] == 3, f"scripted: expected right, got {sa[0]}"
    # agent 1 at (4,6) with goal RIGHT (bc=6): should go up
    assert sa[1] == 0, f"scripted: expected up, got {sa[1]}"

    # test masking
    env_masked = ThreeCorridorVec(num_envs=2, seed=1, mask_partner_obs=True)
    obs_m, _ = env_masked.reset()
    assert obs_m[:, :, 2].sum() == 0, "partner channel should be zeroed when masked"

    print(f"obs {obs.shape}, goal {gv.shape} — env smoke test passed")
