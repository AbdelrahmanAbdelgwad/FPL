"""Tabular agents for the reward-level vs Q-level composition comparison.

The grid is small and discrete, so we can learn exact tabular (fulfillment-)
Q-values and isolate the single variable of interest: **where the two objectives
are composed**.

* :class:`ScalarQAgent` composes at the *reward* level -- it collapses the vector
  reward into a scalar with one of the ``scalarization`` functions and then runs
  ordinary Q-learning. With the geometric composer this reward is identically
  zero, so the agent never receives a gradient.
* :class:`VectorFQAgent` composes at the *Q-value* level -- it keeps one
  normalized fulfillment-Q-value per beacon and composes them with
  :func:`scalarization.q_level_fpl_and` only when choosing actions. Each FQ
  bootstraps from its own sparse reward, so both objectives carry a dense signal
  *before* they are combined.

Both agents key off the integer grid position (from ``info["position"]``) rather
than the normalized Box observation, so the same env serves tabular and
function-approximation agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from scalarization import q_level_fpl_and


def random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    """Argmax with ties broken uniformly (a flat table => true random walk)."""
    m = values.max()
    candidates = np.flatnonzero(values >= m - 1e-12)
    return int(rng.choice(candidates))


@dataclass
class TrainConfig:
    episodes: int = 1500
    gamma: float = 0.95
    lr: float = 0.3
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_frac: float = 0.6
    seed: int = 0

    def epsilon(self, episode: int) -> float:
        decay = max(1, int(self.eps_decay_frac * self.episodes))
        frac = min(1.0, episode / decay)
        return self.eps_start + frac * (self.eps_end - self.eps_start)


def pos_to_state(position, N: int) -> int:
    x, y = position
    return int(x) * N + int(y)


class VectorFQAgent:
    """Q-level composition (FPL / BPG-style).

    Learns ``FQ_k(s, a) = (1 - gamma) * E[discounted fulfillment of beacon k]``
    and acts by ``argmax_a  q_level_fpl_and(FQ(s, a), p)``.

    ``bootstrap``:
      * ``"on_policy"`` backs each objective up through the action chosen by the
        composed greedy policy (faithful to BPG's ``FQ_targ(s', pi(s'))``).
      * ``"per_objective"`` backs each objective up through its own max, i.e.
        learns each beacon's optimal fulfillment field independently and only
        couples them at action-selection time.
    """

    def __init__(self, n_states: int, n_actions: int, n_objectives: int = 2,
                 p: float = 0.0, bootstrap: str = "on_policy"):
        self.fq = np.zeros((n_states, n_actions, n_objectives))
        self.n_actions = n_actions
        self.p = p
        self.bootstrap = bootstrap

    def utilities(self, state: int) -> np.ndarray:
        """Composed utility of every action in ``state``.

        Vectorized equivalent of calling :func:`scalarization.q_level_fpl_and`
        on ``FQ(s, a)`` for each action -- the power-mean conjunction across the
        objective axis.
        """
        fqs = np.clip(self.fq[state], 1e-12, None)  # (n_actions, n_objectives)
        if self.p == 0:
            return np.exp(np.mean(np.log(fqs), axis=1))
        return np.mean(fqs ** self.p, axis=1) ** (1.0 / self.p)

    def greedy_action(self, state: int, rng: np.random.Generator) -> int:
        return random_argmax(self.utilities(state), rng)

    def update(self, s, a, r_vec, s_next, gamma, lr, rng):
        if self.bootstrap == "per_objective":
            boot = self.fq[s_next].max(axis=0)
        else:
            a_next = self.greedy_action(s_next, rng)
            boot = self.fq[s_next, a_next]
        target = (1.0 - gamma) * np.asarray(r_vec, dtype=np.float64) + gamma * boot
        self.fq[s, a] += lr * (target - self.fq[s, a])


class ScalarQAgent:
    """Reward-level composition: scalarize first, then plain Q-learning."""

    def __init__(self, n_states: int, n_actions: int,
                 reward_composer: Callable[[np.ndarray], float]):
        self.q = np.zeros((n_states, n_actions))
        self.n_actions = n_actions
        self.reward_composer = reward_composer

    def greedy_action(self, state: int, rng: np.random.Generator) -> int:
        return random_argmax(self.q[state], rng)

    def update(self, s, a, r_vec, s_next, gamma, lr, rng):
        r = self.reward_composer(r_vec)
        a_next = self.greedy_action(s_next, rng)
        target = (1.0 - gamma) * r + gamma * self.q[s_next, a_next]
        self.q[s, a] += lr * (target - self.q[s, a])
