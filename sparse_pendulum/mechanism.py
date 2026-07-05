"""Why does Q-level composition help as the objective gets sparse? Measure it.

This is a *training-free* probe of the mechanism behind the sparse-Pendulum
result. Claim: composing a binary objective with another at the **reward** level
makes the learning signal vanish as the objective gets sparse (the geomean is 0
unless the binary part fires *this step*), whereas composing at the **Q-value**
level keeps the signal dense, because each objective's value (discounted
return) spreads the rare "upright" events across all the states that lead to
them.

We don't need a trained agent to see this -- we only need the *signal the learner
would receive*:

* reward-level signal  s_r(t) = geomean(angle_binary(t), actuation(t))
* Q-level signal        s_q(t) = geomean( FV_angle(t), FV_actuation(t) )

where ``FV_k(t) = (1-gamma) * sum_{k>=t} gamma^{k-t} r_k`` is the normalized
discounted return-to-go for objective ``k`` -- a Monte-Carlo proxy for the
fulfillment-Q-value a per-objective critic would learn. We roll out an
*exploratory* (uniform-random) policy, exactly the data sitting in an early
replay buffer, and report, as a function of the sparsity knob ``band``:

* the fraction of transitions that carry a non-zero signal, and
* the spatial spread (std) of the signal -- a flat-zero signal has no gradient.

    python sparse_pendulum/mechanism.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (cmorl)
from sparse_pendulum import geomean  # noqa: E402  (also puts envs/Pendulum on sys.path)
import Pendulum  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GAMMA = 0.99


def collect(n_episodes=40, ep_len=200, seed=0):
    """Roll out a uniform-random policy; return per-episode (normed_angle, actuation)."""
    env = Pendulum.PendulumEnv(g=10.0, setpoint=0.0)
    rng = np.random.default_rng(seed)
    low, high = env.action_space.low, env.action_space.high
    episodes = []
    for ep in range(n_episodes):
        env.reset(seed=seed * 1000 + ep)
        nd, act = [], []
        for _ in range(ep_len):
            a = rng.uniform(low, high)
            env.step(a)
            th, _ = env.state
            nd.append(Pendulum.normed_angular_distance(th, env.setpoint))
            act.append(1.0 - (abs(a[0] / env.max_torque)) ** 2.0)
        episodes.append((np.array(nd), np.array(act)))
    return episodes


def discounted_returns(r, gamma=GAMMA):
    """Normalized discounted return-to-go (1-gamma)*sum gamma^k r_{t+k}."""
    v = np.zeros_like(r, dtype=np.float64)
    acc = 0.0
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + gamma * acc
        v[t] = acc
    return (1.0 - gamma) * v


def signals_for_band(episodes, band):
    s_reward, s_q = [], []
    for nd, act in episodes:
        angle_bin = (nd <= band).astype(np.float64)
        # reward-level: geomean of the per-step fulfillments
        s_reward.extend([geomean([angle_bin[t], act[t]]) for t in range(len(act))])
        # Q-level proxy: geomean of the per-objective discounted returns
        fv_a = discounted_returns(angle_bin)
        fv_u = discounted_returns(act)
        s_q.extend([geomean([fv_a[t], fv_u[t]]) for t in range(len(act))])
    return np.array(s_reward), np.array(s_q)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    episodes = collect()
    bands = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

    frac_r, frac_q, std_r, std_q, fire = [], [], [], [], []
    print(f"{'band':>7}{'±deg':>6}{'%buf upright':>13}{'reward sig≠0':>14}{'Q sig≠0':>10}"
          f"{'reward std':>12}{'Q std':>9}")
    for b in bands:
        sr, sq = signals_for_band(episodes, b)
        fire_frac = np.mean([np.mean(((nd <= b)).astype(float)) for nd, _ in episodes])
        fr, fq = np.mean(sr > 1e-6), np.mean(sq > 1e-6)
        dr, dq = float(np.std(sr)), float(np.std(sq))
        frac_r.append(fr); frac_q.append(fq); std_r.append(dr); std_q.append(dq); fire.append(fire_frac)
        print(f"{b:>7}{b*180:>6.0f}{fire_frac*100:>12.1f}%{fr*100:>13.1f}%{fq*100:>9.1f}%{dr:>12.4f}{dq:>9.4f}")

    # Colorblind-safe pair (validated): blue = Q-level, orange = reward-level.
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200, "font.size": 11,
        "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False,
    })
    x = np.array(bands)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    ax1.plot(x, np.array(frac_q) * 100, "o-", color="#2a78d6", lw=2, ms=6,
             mec="white", mew=0.7, label="Q-level signal (returns)")
    ax1.plot(x, np.array(frac_r) * 100, "o-", color="#eb6834", lw=2, ms=6,
             mec="white", mew=0.7, label="reward-level signal")
    ax1.plot(x, np.array(fire) * 100, "s--", color="0.5", lw=1.5, ms=5,
             label="raw '1's in buffer (angle fires)")
    ax1.set_xscale("log"); ax1.invert_xaxis()
    ax1.set_xlabel("band  (sparser →, log scale)")
    ax1.set_ylabel("% of transitions with a non-zero learning signal")
    ax1.set_title("Reward-level signal vanishes with sparsity;\nQ-level stays dense")
    ax1.grid(which="both"); ax1.legend(fontsize=9)

    ratio = np.array(frac_q) / np.maximum(np.array(frac_r), 1e-9)
    ax2.plot(x, ratio, "o-", color="#2a78d6", lw=2, ms=6, mec="white", mew=0.7)
    ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.invert_xaxis()
    ax2.set_xlabel("band  (sparser →, log scale)")
    ax2.set_ylabel("Q-level signal density ÷ reward-level density")
    ax2.set_title("The sparser the objective,\nthe bigger the Q-level advantage")
    ax2.grid(which="both")
    for xi, ri in zip(x, ratio):
        ax2.annotate(f"{ri:.0f}×", (xi, ri), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9, color="0.25")

    fig.suptitle("Mechanism: where does the learning signal go as the objective gets sparse?")
    out = os.path.join(RESULTS_DIR, "mechanism.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
