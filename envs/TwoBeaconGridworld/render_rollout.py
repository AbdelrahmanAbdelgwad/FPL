"""Animated greedy-policy rollouts for TwoBeaconGridworld -> results/rollouts.gif.

Trains each method, then renders a side-by-side GIF of every greedy policy moving
on the grid (agent marker + path trail + live beacon-hit counter), so you can
*watch* the Q-level policy shuttle between both beacons while the reward-level
baselines camp on one (or wander, in the degenerate geomean case).

    python envs/TwoBeaconGridworld/render_rollout.py
    python envs/TwoBeaconGridworld/render_rollout.py --episodes 800 --rollout-steps 60 --seed 0
    python envs/TwoBeaconGridworld/render_rollout.py --methods qlevel_fpl reward_linear
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from two_beacon_gridworld import TwoBeaconGridworldEnv
from tabular_agents import TrainConfig, pos_to_state
from run_experiment import make_agent, train, METHOD_COLORS, RESULTS_DIR

# Short titles so the side-by-side panels stay readable.
SHORT_LABELS = {
    "reward_geometric": "reward geomean\n(== 0, impossible)",
    "reward_smoothed": "reward smoothed\ngeomean",
    "reward_linear": "reward linear",
    "qlevel_fpl": "Q-level FPL\n(decoupled)",
    "qlevel_fpl_onpolicy": "Q-level FPL\n(on-policy/BPG)",
}


def action_values(agent, s):
    """Per-action value driving the greedy policy (composed utility for the
    Q-level agent, the scalar Q-row for the reward-level agents)."""
    if hasattr(agent, "utilities"):
        return np.asarray(agent.utilities(s), dtype=float)
    return np.asarray(agent.q[s], dtype=float)


def greedy_determinism(agent, env):
    """Percent of cells with a single greedy action (the rest are ties broken at
    random). 0% means a flat table -> a pure random walk."""
    uniq = total = 0
    for (x, y) in env._free_cells:
        v = action_values(agent, pos_to_state((x, y), env.N))
        total += 1
        if int(np.sum(v >= v.max() - 1e-12)) == 1:
            uniq += 1
    return 100.0 * uniq / max(total, 1)


def rollout(agent, env, steps, rng):
    """Run the greedy policy and record the path and running beacon-hit tally."""
    _, info = env.reset()
    s = pos_to_state(info["position"], env.N)
    xs, ys, tally = [], [], []
    hits = {1: 0, 2: 0}
    for _ in range(steps):
        x, y = env.position
        rv = info["reward_vec"]
        if rv[0] > 0:
            hits[1] += 1
        if rv[1] > 0:
            hits[2] += 1
        xs.append(int(x))
        ys.append(int(y))
        tally.append((hits[1], hits[2]))
        a = agent.greedy_action(s, rng)
        _, _, _, truncated, info = env.step(a)
        s = pos_to_state(info["position"], env.N)
        if truncated:
            break
    return np.array(xs), np.array(ys), tally


def draw_grid(ax, env):
    """Static background: cell borders, walls, and the two beacons."""
    N = env.N
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)  # invert y so row 0 is on top, matching ansi render
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color="0.85", lw=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for (wx, wy) in env.walls:
        ax.add_patch(plt.Rectangle((wx - 0.5, wy - 0.5), 1, 1, color="0.3"))
    for (bx, by), c in [(env.beacon_1, "1"), (env.beacon_2, "2")]:
        ax.add_patch(plt.Rectangle((bx - 0.5, by - 0.5), 1, 1, color="gold"))
        ax.text(bx, by, c, ha="center", va="center", fontsize=12, fontweight="bold")


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
    parser.add_argument("--methods", nargs="+", default=[
        "reward_geometric", "reward_smoothed", "reward_linear",
        "qlevel_fpl", "qlevel_fpl_onpolicy"])
    parser.add_argument("--rollout-steps", type=int, default=60)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_states = args.N * args.N

    # Train each method and roll out its greedy policy.
    paths = {}
    for name in args.methods:
        env = TwoBeaconGridworldEnv(N=args.N, max_steps=args.max_steps, walls_mode=args.walls_mode)
        cfg = TrainConfig(episodes=args.episodes, gamma=args.gamma, lr=args.lr, seed=args.seed)
        agent = make_agent(name, n_states, args.eps_smooth)
        train(agent, env, cfg, eval_every=10 ** 9)  # train only; skip periodic eval
        det = greedy_determinism(agent, env)
        rng = np.random.default_rng(args.seed + 999)
        xs, ys, tally = rollout(agent, env, args.rollout_steps, rng)
        paths[name] = {"env": env, "xs": xs, "ys": ys, "tally": tally, "det": det}
        print(f"{name:<22} {det:>3.0f}% deterministic   final beacon hits -> "
              f"1:{tally[-1][0]:>2}  2:{tally[-1][1]:>2}  "
              f"({'BOTH' if min(tally[-1]) > 0 else 'one/none'})")

    n_frames = max(len(p["xs"]) for p in paths.values())
    names = args.methods

    fig, axes = plt.subplots(1, len(names), figsize=(2.7 * len(names), 3.4))
    if len(names) == 1:
        axes = [axes]

    artists = {}
    for ax, name in zip(axes, names):
        p = paths[name]
        draw_grid(ax, p["env"])
        color = METHOD_COLORS[name]
        trail, = ax.plot([], [], "-", color=color, lw=1.6, alpha=0.55)
        agent_dot, = ax.plot([], [], "o", color=color, ms=13, mec="black", mew=0.8)
        title = ax.set_title("", fontsize=8)
        artists[name] = (trail, agent_dot, title)

    def update(f):
        changed = []
        for name in names:
            p = paths[name]
            idx = min(f, len(p["xs"]) - 1)
            trail, agent_dot, title = artists[name]
            trail.set_data(p["xs"][:idx + 1], p["ys"][:idx + 1])
            agent_dot.set_data([p["xs"][idx]], [p["ys"][idx]])
            h1, h2 = p["tally"][idx]
            title.set_text(f"{SHORT_LABELS.get(name, name)}  ({p['det']:.0f}% det.)\n"
                           f"beacon hits  1:{h1}  2:{h2}")
            changed += [trail, agent_dot, title]
        return changed

    fig.suptitle(f"Greedy rollout on TwoBeaconGridworld (N={args.N}, step {{}}/{n_frames})", fontsize=11)

    def update_with_step(f):
        fig.suptitle(f"Greedy rollout on TwoBeaconGridworld (N={args.N}, step {f + 1}/{n_frames})",
                     fontsize=11)
        return update(f)

    fig.text(0.5, 0.01,
             "Greedy & deterministic given the seed; ties (equal-value actions) are broken at "
             "random. 0% det. = flat table = pure random walk.",
             ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    anim = FuncAnimation(fig, update_with_step, frames=n_frames, blit=False)
    out = args.out or os.path.join(RESULTS_DIR, "rollouts.gif")
    anim.save(out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"\nSaved animation to {out}")


if __name__ == "__main__":
    main()
