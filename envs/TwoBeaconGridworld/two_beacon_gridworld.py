"""TwoBeaconGridworld: a minimal env for testing Q-level objective composition.

The environment exposes **two non-overlapping sparse binary reward components**.
Because the two beacons occupy distinct cells, the agent can never be on both at
once, so for every transition::

    reward[0] * reward[1] == 0.0

This makes the *reward-level* conjunction (a geometric / power mean of the two
binary signals) identically zero -- there is nothing to optimize. The whole
point of the env is to contrast that with composing the same two objectives at
the **Q-value / fulfillment-Q-value level**, where each binary reward turns into
a dense, continuous expected-return signal that *can* be composed and optimized.

Coordinate convention
---------------------
A position is ``(x, y)`` with ``x`` the column (horizontal, 0..N-1) and ``y`` the
row (vertical, 0..N-1, increasing downward like image coordinates). Discrete
actions are ``0=up (y-1)``, ``1=down (y+1)``, ``2=left (x-1)``, ``3=right (x+1)``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# discrete action index -> (dx, dy)
_DISCRETE_DELTAS: tuple[tuple[int, int], ...] = (
    (0, -1),  # 0: up
    (0, 1),   # 1: down
    (-1, 0),  # 2: left
    (1, 0),   # 3: right
)


class TwoBeaconGridworldEnv(gym.Env):
    """Two-beacon sparse multi-objective grid world.

    Parameters
    ----------
    N:
        Grid side length (``N x N``). Default ``15``.
    max_steps:
        Episode horizon; the episode truncates once ``step_count >= max_steps``.
    random_start:
        If ``True`` the agent starts at a uniformly random non-wall cell,
        otherwise at the grid center ``(N // 2, N // 2)``.
    action_mode:
        ``"discrete"`` (``Discrete(4)``) or ``"continuous"`` (``Box(2,)``).
    walls_mode:
        ``"none"`` (open grid) or ``"four_rooms"`` (four rooms joined by
        one-cell doorways). Beacons and start are guaranteed reachable.
    render_mode:
        ``None`` or ``"ansi"``.
    """

    metadata = {"render_modes": [None, "ansi"]}

    def __init__(
        self,
        N: int = 15,
        max_steps: int = 200,
        random_start: bool = False,
        action_mode: str = "discrete",
        walls_mode: str = "none",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if N < 5:
            raise ValueError("N must be >= 5 so the four-rooms layout and beacons fit")
        if action_mode not in ("discrete", "continuous"):
            raise ValueError(f"unknown action_mode {action_mode!r}")
        if walls_mode not in ("none", "four_rooms"):
            raise ValueError(f"unknown walls_mode {walls_mode!r}")

        self.N = N
        self.max_steps = max_steps
        self.random_start = random_start
        self.action_mode = action_mode
        self.walls_mode = walls_mode
        self.render_mode = render_mode

        self.beacon_1 = (1, 1)
        self.beacon_2 = (N - 2, N - 2)
        self.center = (N // 2, N // 2)

        self.walls = self._build_walls()
        self._free_cells = [
            (x, y) for x in range(N) for y in range(N) if (x, y) not in self.walls
        ]

        # Observation: normalized (x, y) in [0, 1]^2.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        # Vector reward space, one binary fulfillment per beacon.
        self.reward_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        if action_mode == "discrete":
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.position = np.array(self.center, dtype=int)
        self.step_count = 0

    # -- layout -------------------------------------------------------------
    def _build_walls(self) -> frozenset[tuple[int, int]]:
        if self.walls_mode == "none":
            return frozenset()

        N = self.N
        mid = N // 2
        walls: set[tuple[int, int]] = set()
        # A cross of walls splits the grid into four rooms.
        for i in range(N):
            walls.add((mid, i))
            walls.add((i, mid))
        # One-cell doorways: the midpoint of each of the four wall arms.
        doorways = {
            (mid, mid // 2),
            (mid, mid + 1 + (N - 1 - mid) // 2),
            (mid // 2, mid),
            (mid + 1 + (N - 1 - mid) // 2, mid),
        }
        walls -= doorways
        # Keep the center (start) and its four neighbours open so the start cell
        # is a small junction connecting all four rooms; never wall the beacons.
        cx, cy = self.center
        always_open = {
            (cx, cy), (cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1),
            self.beacon_1, self.beacon_2,
        }
        walls -= always_open
        return frozenset(w for w in walls if self._in_bounds(w))

    def _in_bounds(self, pos) -> bool:
        x, y = pos
        return 0 <= x < self.N and 0 <= y < self.N

    # -- helpers ------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        x, y = self.position
        return np.array([x / (self.N - 1), y / (self.N - 1)], dtype=np.float32)

    def _reward_vector(self, pos) -> np.ndarray:
        p = (int(pos[0]), int(pos[1]))
        r1 = 1.0 if p == self.beacon_1 else 0.0
        r2 = 1.0 if p == self.beacon_2 else 0.0
        return np.array([r1, r2], dtype=np.float32)

    def _info(self, reward_vec: np.ndarray) -> dict:
        return {
            "position": (int(self.position[0]), int(self.position[1])),
            "beacon_1": self.beacon_1,
            "beacon_2": self.beacon_2,
            "reward_vec": reward_vec,
            "step_count": self.step_count,
        }

    def _decode_action(self, action) -> tuple[int, int]:
        if self.action_mode == "discrete":
            return _DISCRETE_DELTAS[int(action)]
        # continuous: move one cell along the dominant axis
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        ax, ay = a[0], a[1]
        if abs(ax) >= abs(ay):
            return (int(np.sign(ax)), 0) if abs(ax) >= 1e-6 else (0, 0)
        return (0, int(np.sign(ay))) if abs(ay) >= 1e-6 else (0, 0)

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.random_start:
            idx = self.np_random.integers(len(self._free_cells))
            self.position = np.array(self._free_cells[idx], dtype=int)
        else:
            self.position = np.array(self.center, dtype=int)
        self.step_count = 0
        reward_vec = self._reward_vector(self.position)
        return self._get_obs(), self._info(reward_vec)

    def step(self, action):
        dx, dy = self._decode_action(action)
        nx = self.position[0] + dx
        ny = self.position[1] + dy
        candidate = (int(nx), int(ny))
        # Boundary collisions and walls leave the agent in place.
        if self._in_bounds(candidate) and candidate not in self.walls:
            self.position = np.array(candidate, dtype=int)

        self.step_count += 1
        reward_vec = self._reward_vector(self.position)
        terminated = False
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), reward_vec, terminated, truncated, self._info(reward_vec)

    def render(self):
        if self.render_mode != "ansi":
            return None
        rows = []
        ax, ay = int(self.position[0]), int(self.position[1])
        for y in range(self.N):
            chars = []
            for x in range(self.N):
                if (x, y) == (ax, ay):
                    chars.append("A")
                elif (x, y) == self.beacon_1:
                    chars.append("1")
                elif (x, y) == self.beacon_2:
                    chars.append("2")
                elif (x, y) in self.walls:
                    chars.append("#")
                else:
                    chars.append(".")
            rows.append(" ".join(chars))
        return "\n".join(rows)


gym.register("TwoBeaconGridworld-v0", TwoBeaconGridworldEnv)  # type: ignore
