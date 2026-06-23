"""Scalarization utilities for the TwoBeaconGridworld experiment.

These map a vector of values (either the raw binary reward vector or a vector of
fulfillment-Q-values) into a single scalar. They are the knobs that decide
*where* and *how* the two objectives are composed:

* :func:`reward_level_geometric` collapses the binary reward with a geometric
  mean. On this env it is **identically zero** (the two 1s never overlap), which
  is exactly the degenerate, un-optimizable case.
* :func:`reward_level_smoothed_geometric` is the "never output exactly zero"
  trick -- smoothing each binary reward off of 0 before the geometric mean.
* :func:`reward_level_linear` is the classic linear scalarization baseline.
* :func:`q_level_fpl_and` is the FPL conjunction applied at the Q-value level,
  where each objective's Q-value is already a dense signal in [0, 1].
"""

from __future__ import annotations

import numpy as np


def power_mean(x, p, eps: float = 1e-12) -> float:
    """Generalized (power) mean of a non-negative vector ``x``.

    ``p == 0`` -> geometric mean, ``p == -inf`` -> min, ``p == +inf`` -> max,
    otherwise ``(mean(x**p))**(1/p)``. ``eps`` floors the values to keep ``log``
    and negative powers finite.
    """
    x = np.asarray(x, dtype=np.float64)
    if np.isneginf(p):
        return float(np.min(x))
    if np.isposinf(p):
        return float(np.max(x))
    xc = np.clip(x, eps, None)
    if p == 0:
        return float(np.exp(np.mean(np.log(xc))))
    return float(np.mean(xc ** p) ** (1.0 / p))


def reward_level_geometric(reward_vec) -> float:
    """sqrt(r1 * r2). Exactly zero on TwoBeaconGridworld (non-overlapping 1s)."""
    r = np.asarray(reward_vec, dtype=np.float64)
    return float(np.sqrt(r[0] * r[1]))


def reward_level_smoothed_geometric(reward_vec, epsilon: float) -> float:
    """Geometric mean after smoothing each binary reward off of zero.

    ``r_tilde = epsilon + (1 - epsilon) * reward_vec`` so that a single fulfilled
    beacon yields a positive scalar -- the "never output exactly zero" baseline.
    """
    r = np.asarray(reward_vec, dtype=np.float64)
    r_tilde = epsilon + (1.0 - epsilon) * r
    return float(np.sqrt(r_tilde[0] * r_tilde[1]))


def reward_level_linear(reward_vec) -> float:
    """0.5 * (r1 + r2) -- linear scalarization."""
    r = np.asarray(reward_vec, dtype=np.float64)
    return float(0.5 * (r[0] + r[1]))


def q_level_fpl_and(fq_vec, p: float = 0) -> float:
    """FPL conjunction at the Q-value level.

    ``fq_vec`` is a vector of fulfillment-Q-values in [0, 1]. For ``p == 0`` this
    is ``sqrt(fq1 * fq2)``; otherwise the power mean with the given ``p``. This
    only scalarizes the (already computed) Q-values -- it does not compute them.
    """
    fq = np.asarray(fq_vec, dtype=np.float64)
    if p == 0:
        return float(np.sqrt(fq[0] * fq[1]))
    return power_mean(fq, p)
