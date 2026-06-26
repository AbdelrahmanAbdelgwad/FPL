"""Sparse-objective Pendulum: the building blocks for the reward-level vs
Q-level composition comparison.

The whole point of this experiment is to test the chat/paper claim *in the regime
where it should actually matter*: a **gradient-based** learner (BPG/DDPG actor)
optimizing **simultaneous, competing** objectives where one objective is **sparse
/ binary**.

Two competing fulfillments on the classic pendulum:

* ``angle``     -- a *binary* "is the pole upright?" signal: ``1`` iff the
  normalized angle error is within ``band`` of the setpoint, else ``0``. Shrinking
  ``band`` makes it sparser (Bassel's "0 on the lower half, 0--1 near the top").
* ``actuation`` -- a *dense* "use little torque" signal in ``[0, 1]``.

These compete: staying upright needs torque (which lowers ``actuation``), and the
upright signal is binary, so most of the state space gives no angle gradient at
all until the agent is already near the top.

We compose the two with the **geometric mean** (logical AND, ``p = 0``) and vary
only *where* that composition happens:

* **reward level** -- scalarize ``geomean(angle, actuation)`` into the env's
  reward, then train scalar DDPG (``cmorl=None``). With a small ``band`` this
  reward is ``0`` almost everywhere -> almost no gradient.
* **Q-value level (BPG)** -- keep a fulfillment-Q per objective and compose with
  the same geomean only in the actor loss (``CMORL(...)`` + ``p_objectives=0``).
  Each FQ bootstraps from its own signal, so ``angle`` is dense at the Q level
  even though its reward is binary.
"""

from __future__ import annotations

import os
import sys
from functools import partial

import numpy as np
import gymnasium as gym

# Import the Pendulum env module directly (avoid envs/__init__, which pulls in
# TensorFlow-heavy Boids/Bittle deps we don't need here).
_PENDULUM_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "envs", "Pendulum")
if _PENDULUM_DIR not in sys.path:
    sys.path.insert(0, _PENDULUM_DIR)
import Pendulum  # noqa: E402  (envs/Pendulum/Pendulum.py)


def pendulum_fulfillments(action, env, band: float) -> np.ndarray:
    """The two competing fulfillments in ``[0, 1]`` for the current env state.

    ``angle`` is *binary* (1 iff within ``band`` of upright); ``actuation`` is the
    dense low-torque reward. ``band`` is the sparsity knob: smaller = sparser.
    """
    u = np.squeeze(action)
    th, _thdot = env.state
    normed = Pendulum.normed_angular_distance(th, env.setpoint)  # 0 upright .. 1 down
    angle = 1.0 if normed <= band else 0.0
    actuation = 1.0 - (abs(u / env.max_torque)) ** 2.0
    return np.array([angle, actuation], dtype=np.float32)


def sparse_reward_fn(transition, env, band: float) -> np.ndarray:
    """CMORL reward function (vector) for the Q-level arm."""
    return pendulum_fulfillments(transition.action, env, band)


def geomean(vec: np.ndarray, slack: float = 0.0) -> float:
    """Geometric mean of a non-negative vector. ``slack=0`` is the exact geomean
    (a single 0 drives it to 0 -- the degenerate sparse case); ``slack>0`` is the
    'never output exactly zero' smoothing."""
    v = np.asarray(vec, dtype=np.float64) + slack
    with np.errstate(divide="ignore"):  # log(0) -> -inf -> exp -> 0 (the degenerate case)
        return float(np.exp(np.mean(np.log(v))) - slack)


class RewardLevelWrapper(gym.Wrapper):
    """Makes the env return ``geomean(angle, actuation)`` as its scalar reward, so
    plain DDPG (``cmorl=None``) learns the **reward-level** composition.

    ``slack=0`` -> the pure (degenerate) geomean; ``slack>0`` -> the smoothed
    variant.
    """

    def __init__(self, env, band: float, slack: float = 0.0):
        super().__init__(env)
        self.band = band
        self.slack = slack

    def step(self, action):
        obs, _orig_reward, terminated, truncated, info = self.env.step(action)
        fulfillments = pendulum_fulfillments(action, self.env, self.band)
        reward = geomean(fulfillments, slack=self.slack)
        info = {**info, "fulfillments": fulfillments}
        return obs, reward, terminated, truncated, info


def evaluate_policy(pi_network, band: float, n_steps: int = 400, seed: int = 12345) -> dict:
    """Roll out the deterministic policy and measure the *true* objectives,
    identically for every arm: fraction of time upright (the sparse goal), mean
    actuation, and their geomean (the intended AND)."""
    env = Pendulum.PendulumEnv(g=10.0, setpoint=0.0)
    o, _ = env.reset(seed=seed)
    low, high = env.action_space.low, env.action_space.high
    angles, actuations = [], []
    for _ in range(n_steps):
        pi = np.asarray(pi_network(np.array([o], dtype=np.float32)))[0]  # in [-1, 1]
        a = np.clip(pi * (high - low) / 2.0 + (high + low) / 2.0, low, high)
        o, _, terminated, truncated, _ = env.step(a)
        f = pendulum_fulfillments(a, env, band)
        angles.append(f[0])
        actuations.append(f[1])
        if terminated or truncated:
            o, _ = env.reset()
    angle_frac = float(np.mean(angles))      # fraction of time upright
    actuation_mean = float(np.mean(actuations))
    return {
        "angle_frac": angle_frac,
        "actuation": actuation_mean,
        "geomean": float(np.sqrt(max(angle_frac, 0.0) * max(actuation_mean, 0.0))),
    }


def make_cmorl(band: float):
    """CMORL spec for the Q-level arm (vector reward; default geomean composer is
    selected via ``p_objectives=0`` in the hyperparameters)."""
    from cmorl.utils.reward_utils import CMORL
    return CMORL(partial(sparse_reward_fn, band=band))
