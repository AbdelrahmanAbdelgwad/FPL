"""Render each method's learned greedy policy as an arrow map -> results/policies.png.

Unlike a rollout, this shows the *whole* deterministic policy at once: one arrow
per cell for the unique greedy action, and a marked dot where several actions tie
(so the greedy choice there is broken at random). It makes clear that the
reward-geomean "policy" is one big tie field (a flat, all-zero table => a pure
random walk), while the Q-level policy is a structured flow into both beacons.

    python envs/TwoBeaconGridworld/policy_map.py
    python envs/TwoBeaconGridworld/policy_map.py --episodes 800 --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from two_beacon_gridworld import TwoBeaconGridworldEnv, _DISCRETE_DELTAS
from tabular_agents import TrainConfig, pos_to_state
from run_experiment import make_agent, train, METHOD_COLORS, RESULTS_DIR
from render_rollout import draw_grid, SHORT_LABELS


def action_values(agent, s):
    """Per-action value used by the greedy policy (composed utility for the
    Q-level agent, the scalar Q-row for the reward-level agents)."""
    if hasattr(agent, "utilities"):
        return np.asarray(agent.utilities(s), dtype=float)
    return np.asarray(agent.q[s], dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--eps-smooth", type=float, default=1e-3)
    parser.add_argument("--walls-mode", choices=["none", "four_rooms"], default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=[
        "reward_geometric", "reward_smoothed", "reward_linear",
        "qlevel_fpl", "qlevel_fpl_onpolicy"])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    N = args.N

    fig, axes = plt.subplots(1, len(args.methods), figsize=(2.9 * len(args.methods), 3.6))
    if len(args.methods) == 1:
        axes = [axes]

    for ax, name in zip(axes, args.methods):
        env = TwoBeaconGridworldEnv(N=N, max_steps=args.max_steps, walls_mode=args.walls_mode)
        agent = make_agent(name, N * N, args.eps_smooth)
        train(agent, env, TrainConfig(episodes=args.episodes, gamma=args.gamma,
                                      lr=args.lr, seed=args.seed), eval_every=10 ** 9)
        draw_grid(ax, env)
        color = METHOD_COLORS[name]

        unique = 0
        total = 0
        axc, ayc, U, V = [], [], [], []
        for (x, y) in env._free_cells:
            s = pos_to_state((x, y), N)
            v = action_values(agent, s)
            m = v.max()
            greedy = np.flatnonzero(v >= m - 1e-12)
            total += 1
            if len(greedy) == 1:
                unique += 1
                dx, dy = _DISCRETE_DELTAS[int(greedy[0])]
                axc.append(x); ayc.append(y); U.append(dx); V.append(dy)
            else:
                # tie -> the greedy action here is chosen at random
                ax.scatter([x], [y], s=70, facecolors="none", edgecolors=color, lw=1.6, zorder=3)
                ax.text(x, y + 0.02, str(len(greedy)), ha="center", va="center",
                        fontsize=7, color=color, zorder=4)
        # y axis is inverted (origin upper), so (dx, dy) point the intuitive way.
        ax.quiver(axc, ayc, U, V, color=color, angles="xy", scale_units="xy",
                  scale=2.2, width=0.012, zorder=3)
        # mark the start cell
        sx, sy = env.center
        ax.scatter([sx], [sy], marker="*", s=130, color="black", zorder=5)

        det = 100.0 * unique / max(total, 1)
        ax.set_title(f"{SHORT_LABELS.get(name, name)}\n{det:.0f}% deterministic", fontsize=8)

    fig.suptitle("Learned greedy policy (arrow = unique action;  circle = tie -> random;  "
                 "★ = start)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = args.out or os.path.join(RESULTS_DIR, "policies.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved policy map to {out}")


if __name__ == "__main__":
    main()
