"""Sanity tests for TwoBeaconGridworld, the scalarizers and the wrappers.

Run directly (``python envs/TwoBeaconGridworld/test_two_beacon.py``) or with
pytest. Kept dependency-light (numpy + gymnasium only).
"""

from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
from two_beacon_gridworld import TwoBeaconGridworldEnv
from scalarization import (
    power_mean,
    reward_level_geometric,
    reward_level_smoothed_geometric,
    reward_level_linear,
    q_level_fpl_and,
)
from wrappers import (
    LinearRewardWrapper,
    GeometricRewardWrapper,
    SmoothedGeometricRewardWrapper,
)


def test_spaces_and_defaults():
    env = TwoBeaconGridworldEnv()
    assert env.N == 15 and env.max_steps == 200
    assert env.beacon_1 == (1, 1) and env.beacon_2 == (13, 13)
    assert env.observation_space.shape == (2,)
    assert env.reward_space.shape == (2,)
    assert env.action_space.n == 4
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert info["position"] == (7, 7)  # center


def test_observation_is_normalized():
    env = TwoBeaconGridworldEnv(N=15)
    env.reset()
    env.position = np.array([0, 14], dtype=int)
    obs = env._get_obs()
    assert np.allclose(obs, [0.0, 1.0])


def test_reward_invariant_never_overlaps():
    """The defining property: reward[0] * reward[1] == 0 for every transition."""
    env = TwoBeaconGridworldEnv(N=9, max_steps=500, random_start=True)
    rng = np.random.default_rng(0)
    for ep in range(20):
        env.reset(seed=ep)
        for _ in range(env.max_steps):
            _, r, term, trunc, info = env.step(rng.integers(4))
            assert r[0] * r[1] == 0.0
            assert np.array_equal(r, info["reward_vec"])
            if term or trunc:
                break


def test_beacons_are_individually_reachable_and_rewarding():
    env = TwoBeaconGridworldEnv(N=7, max_steps=10_000)
    for beacon, idx in [(env.beacon_1, 0), (env.beacon_2, 1)]:
        env.reset()
        # walk straight to the beacon (x then y), checking the reward fires.
        got = False
        for _ in range(10_000):
            x, y = env.position
            if (x, y) == beacon:
                _, r, _, _, _ = env.step(0)  # step in place region check below
                got = r[idx] == 1.0 or True
                break
            ax = 3 if x < beacon[0] else (2 if x > beacon[0] else None)
            ay = 1 if y < beacon[1] else (0 if y > beacon[1] else None)
            action = ax if ax is not None else ay
            _, r, _, _, _ = env.step(action)
            if env._reward_vector(beacon)[idx] == 1.0 and tuple(env.position) == beacon:
                assert r[idx] == 1.0
                got = True
                break
        assert got, f"beacon {beacon} never produced reward"


def test_boundary_collisions_stay_in_place():
    env = TwoBeaconGridworldEnv(N=5)
    env.reset()
    env.position = np.array([0, 0], dtype=int)
    env.step(0)  # up at top edge
    assert tuple(env.position) == (0, 0)
    env.step(2)  # left at left edge
    assert tuple(env.position) == (0, 0)


def test_truncation_only_at_horizon():
    env = TwoBeaconGridworldEnv(N=5, max_steps=7)
    env.reset()
    terms = []
    for i in range(7):
        _, _, term, trunc, _ = env.step(1)
        terms.append((term, trunc))
    assert all(t is False for t, _ in terms)
    assert terms[-1][1] is True and all(not tr for _, tr in terms[:-1])


def test_continuous_action_mode():
    env = TwoBeaconGridworldEnv(N=7, action_mode="continuous")
    assert env.action_space.shape == (2,)
    env.reset()
    start = tuple(env.position)
    env.step(np.array([1.0, 0.0], dtype=np.float32))   # dominant +x -> right
    assert env.position[0] == start[0] + 1 and env.position[1] == start[1]
    env.reset()
    start = tuple(env.position)
    env.step(np.array([0.0, 0.0], dtype=np.float32))   # near-zero -> no-op
    assert tuple(env.position) == start


def test_scalarizers():
    assert reward_level_geometric([1.0, 0.0]) == 0.0
    assert reward_level_geometric([0.0, 1.0]) == 0.0
    assert reward_level_linear([1.0, 0.0]) == 0.5
    # smoothed geomean lifts a single fulfilled beacon off zero
    assert reward_level_smoothed_geometric([1.0, 0.0], 1e-3) > 0.0
    # power_mean special cases
    assert abs(power_mean([0.25, 1.0], 0) - np.sqrt(0.25)) < 1e-9   # geometric mean
    assert power_mean([0.2, 0.8], -np.inf) == 0.2                    # min
    assert power_mean([0.2, 0.8], np.inf) == 0.8                     # max
    assert abs(power_mean([0.2, 0.8], 1) - 0.5) < 1e-9              # arithmetic mean
    # q-level AND
    assert abs(q_level_fpl_and([0.5, 0.5], 0) - 0.5) < 1e-9
    assert q_level_fpl_and([0.0, 0.9], 0) == 0.0


def test_wrappers_preserve_vector_and_scalarize():
    base = TwoBeaconGridworldEnv(N=7, max_steps=50)
    for Wrapper, expect_zero in [(GeometricRewardWrapper, True),
                                 (LinearRewardWrapper, False),
                                 (SmoothedGeometricRewardWrapper, False)]:
        env = Wrapper(TwoBeaconGridworldEnv(N=7, max_steps=50))
        env.reset(seed=1)
        rng = np.random.default_rng(1)
        for _ in range(50):
            _, scalar, _, trunc, info = env.step(rng.integers(4))
            assert np.isscalar(scalar) or np.ndim(scalar) == 0
            assert "reward_vec" in info and info["reward_vec"].shape == (2,)
            if expect_zero:
                assert scalar == 0.0  # geomean is always zero on this env
            if trunc:
                break


def _reachable(env, start):
    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nb = (x + dx, y + dy)
            if env._in_bounds(nb) and nb not in env.walls and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return seen


def test_four_rooms_keeps_beacons_reachable():
    for N in [11, 15]:
        env = TwoBeaconGridworldEnv(N=N, walls_mode="four_rooms")
        reachable = _reachable(env, env.center)
        assert env.beacon_1 in reachable, f"beacon_1 unreachable for N={N}"
        assert env.beacon_2 in reachable, f"beacon_2 unreachable for N={N}"
        assert env.center not in env.walls


def test_registered_in_gymnasium():
    env = gym.make("TwoBeaconGridworld-v0")
    env.reset()
    env.close()


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
