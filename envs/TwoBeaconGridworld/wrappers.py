"""Reward wrappers that scalarize TwoBeaconGridworld's vector reward.

Each wrapper turns the env's vector reward into a single scalar (so the env can
be driven by an ordinary single-objective algorithm) while preserving the
original vector reward in ``info["reward_vec"]``.

``GeometricRewardWrapper`` always emits ``0.0`` on this env, because the two
sparse rewards never overlap -- that is the failure case these experiments exist
to expose.
"""

from __future__ import annotations

import gymnasium as gym

from scalarization import (
    reward_level_linear,
    reward_level_geometric,
    reward_level_smoothed_geometric,
)


class _ScalarizeWrapper(gym.Wrapper):
    """Base class: scalarize the vector reward, keep the vector in info."""

    def _scalarize(self, reward_vec) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def step(self, action):
        obs, reward_vec, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["reward_vec"] = reward_vec
        scalar = self._scalarize(reward_vec)
        return obs, scalar, terminated, truncated, info


class LinearRewardWrapper(_ScalarizeWrapper):
    """Scalar reward = 0.5 * (r1 + r2)."""

    def _scalarize(self, reward_vec) -> float:
        return reward_level_linear(reward_vec)


class GeometricRewardWrapper(_ScalarizeWrapper):
    """Scalar reward = sqrt(r1 * r2); always 0.0 on TwoBeaconGridworld."""

    def _scalarize(self, reward_vec) -> float:
        return reward_level_geometric(reward_vec)


class SmoothedGeometricRewardWrapper(_ScalarizeWrapper):
    """Scalar reward = smoothed geometric mean (the "never output 0" baseline)."""

    def __init__(self, env, epsilon: float = 1e-3):
        super().__init__(env)
        self.epsilon = epsilon

    def _scalarize(self, reward_vec) -> float:
        return reward_level_smoothed_geometric(reward_vec, self.epsilon)
