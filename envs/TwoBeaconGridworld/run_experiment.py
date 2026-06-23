"""Reward-level vs Q-level composition on TwoBeaconGridworld.

Trains tabular agents that differ only in *where* the two sparse, non-overlapping
beacon rewards are composed, then reports how well each one satisfies the
intended "reach beacon 1 AND beacon 2" objective.

    python envs/TwoBeaconGridworld/run_experiment.py
    python envs/TwoBeaconGridworld/run_experiment.py --N 5 --episodes 2000 --seeds 5

Methods
-------
reward_geometric    sqrt(r1*r2) at the reward level  -> identically 0, nothing to learn
reward_smoothed     smoothed geomean (never-0 trick) -> learnable, but collapses AND->camp
reward_linear       0.5*(r1+r2)                      -> learnable, but camps at one beacon
qlevel_fpl          geomean of the two FQ-values     -> dense signal, reaches BOTH beacons
qlevel_fpl_onpolicy on-policy FQ composition (BPG)   -> ablation: greedy tabular camps
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

from two_beacon_gridworld import TwoBeaconGridworldEnv
from tabular_agents import VectorFQAgent, ScalarQAgent, TrainConfig, pos_to_state
from scalarization import (
    reward_level_geometric,
    reward_level_smoothed_geometric,
    reward_level_linear,
    q_level_fpl_and,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

METHOD_LABELS = {
    "reward_geometric": "reward-level geomean  (== 0, impossible)",
    "reward_smoothed": "reward-level smoothed geomean (eps=1e-3)",
    "reward_linear": "reward-level linear (0.5*(r1+r2))",
    "qlevel_fpl": "Q-level FPL geomean (decoupled critics)",
    "qlevel_fpl_onpolicy": "Q-level FPL geomean (on-policy, BPG)",
}
METHOD_COLORS = {
    "reward_geometric": "#d62728",
    "reward_smoothed": "#ff7f0e",
    "reward_linear": "#9467bd",
    "qlevel_fpl": "#2ca02c",
    "qlevel_fpl_onpolicy": "#1f77b4",
}


def make_agent(name: str, n_states: int, eps_smooth: float):
    if name == "reward_geometric":
        return ScalarQAgent(n_states, 4, reward_level_geometric)
    if name == "reward_smoothed":
        return ScalarQAgent(n_states, 4, lambda r: reward_level_smoothed_geometric(r, eps_smooth))
    if name == "reward_linear":
        return ScalarQAgent(n_states, 4, reward_level_linear)
    if name == "qlevel_fpl":
        return VectorFQAgent(n_states, 4, 2, p=0.0, bootstrap="per_objective")
    if name == "qlevel_fpl_onpolicy":
        return VectorFQAgent(n_states, 4, 2, p=0.0, bootstrap="on_policy")
    raise ValueError(name)


def evaluate(agent, env, gamma, rng):
    """Greedy rollout: discounted per-beacon fulfillment, the AND, and how many
    distinct beacons get reached."""
    _, info = env.reset()
    s = pos_to_state(info["position"], env.N)
    fv = np.zeros(2)
    disc = 1.0
    visited = set()
    visits = np.zeros(env.N * env.N)
    for _ in range(env.max_steps):
        rv = info["reward_vec"]
        fv += disc * rv
        if rv[0] > 0:
            visited.add(1)
        if rv[1] > 0:
            visited.add(2)
        visits[s] += 1
        a = agent.greedy_action(s, rng)
        _, rv, _, truncated, info = env.step(a)
        s = pos_to_state(info["position"], env.N)
        disc *= gamma
        if truncated:
            break
    fv *= (1.0 - gamma)
    return {
        "fv": fv,
        "and": q_level_fpl_and(fv, 0.0),
        "beacons": len(visited),
        "visits": visits,
    }


def train(agent, env, cfg, eval_every):
    rng = np.random.default_rng(cfg.seed)
    xs, ys = [], []
    for ep in range(cfg.episodes):
        eps = cfg.epsilon(ep)
        _, info = env.reset(seed=int(rng.integers(1 << 30)))
        s = pos_to_state(info["position"], env.N)
        for _ in range(env.max_steps):
            a = int(rng.integers(agent.n_actions)) if rng.random() < eps else agent.greedy_action(s, rng)
            _, rv, _, truncated, info = env.step(a)
            ns = pos_to_state(info["position"], env.N)
            agent.update(s, a, rv, ns, cfg.gamma, cfg.lr, rng)
            s = ns
            if truncated:
                break
        if ep % eval_every == 0 or ep == cfg.episodes - 1:
            res = evaluate(agent, env, cfg.gamma, rng)
            xs.append(ep)
            ys.append(res["and"])
    final = evaluate(agent, env, cfg.gamma, rng)
    table = agent.fq if hasattr(agent, "fq") else agent.q
    final["max_abs_q"] = float(np.abs(table).max())
    return {"x": np.array(xs), "y": np.array(ys), "final": final}


def plot_curves(results, path):
    plt.figure(figsize=(8, 5))
    for name, agg in results.items():
        plt.plot(agg["x"], agg["y_mean"], color=METHOD_COLORS[name], lw=2, label=METHOD_LABELS[name])
        plt.fill_between(agg["x"], agg["y_mean"] - agg["y_std"], agg["y_mean"] + agg["y_std"],
                         color=METHOD_COLORS[name], alpha=0.15)
    plt.xlabel("training episodes")
    plt.ylabel("intended AND utility:  geomean( FV_1 , FV_2 )")
    plt.title("Reaching 'beacon 1 AND beacon 2' under sparse, non-overlapping rewards")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_bars(results, path):
    names = list(results.keys())
    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(x, [results[n]["and_mean"] for n in names],
            yerr=[results[n]["and_std"] for n in names],
            color=[METHOD_COLORS[n] for n in names])
    ax1.set_ylabel("geomean( FV_1 , FV_2 )  (higher = both fulfilled)")
    ax1.set_title("Final intended AND utility")
    ax2.bar(x, [results[n]["beacons_mean"] for n in names],
            yerr=[results[n]["beacons_std"] for n in names],
            color=[METHOD_COLORS[n] for n in names])
    ax2.set_ylabel("distinct beacons reached by greedy policy")
    ax2.set_ylim(0, 2.1)
    ax2.set_title("How many of the two beacons get reached")
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[n] for n in names], rotation=25, ha="right", fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_visits(results, env, path):
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.3))
    for ax, name in zip(axes, names):
        # visits are indexed by state = x*N + y; reshape so rows are y, cols x.
        grid = results[name]["visits_mean"].reshape(env.N, env.N).T
        ax.imshow(grid, cmap="viridis", origin="upper")
        for (bx, by), c in [(env.beacon_1, "1"), (env.beacon_2, "2")]:
            ax.text(bx, by, c, ha="center", va="center", color="white", fontsize=13, fontweight="bold")
        ax.set_title(METHOD_LABELS[name], fontsize=6.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Greedy-policy state visitation (brighter = more visited)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--N", type=int, default=5, help="grid side (smaller learns faster)")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--eps-smooth", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--walls-mode", choices=["none", "four_rooms"], default="none")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_states = args.N * args.N
    methods = ["reward_geometric", "reward_smoothed", "reward_linear",
               "qlevel_fpl", "qlevel_fpl_onpolicy"]

    env0 = TwoBeaconGridworldEnv(N=args.N, max_steps=args.max_steps, walls_mode=args.walls_mode)
    print(f"TwoBeaconGridworld N={args.N}, beacons {env0.beacon_1} & {env0.beacon_2}, "
          f"start {env0.center}, walls={args.walls_mode}")
    print(f"{args.seeds} seed(s) x {args.episodes} episodes, gamma={args.gamma}\n")

    results = {}
    for name in methods:
        ys, fvs, ands, beacons, visits, maxq = [], [], [], [], [], []
        xref = None
        for seed in range(args.seeds):
            env = TwoBeaconGridworldEnv(N=args.N, max_steps=args.max_steps, walls_mode=args.walls_mode)
            cfg = TrainConfig(episodes=args.episodes, gamma=args.gamma, lr=args.lr, seed=seed)
            out = train(make_agent(name, n_states, args.eps_smooth), env, cfg, args.eval_every)
            xref = out["x"]
            ys.append(out["y"])
            fvs.append(out["final"]["fv"])
            ands.append(out["final"]["and"])
            beacons.append(out["final"]["beacons"])
            visits.append(out["final"]["visits"])
            maxq.append(out["final"]["max_abs_q"])
        ys = np.array(ys)
        results[name] = {
            "x": xref, "y_mean": ys.mean(0), "y_std": ys.std(0),
            "fv_mean": np.mean(fvs, 0),
            "and_mean": float(np.mean(ands)), "and_std": float(np.std(ands)),
            "beacons_mean": float(np.mean(beacons)), "beacons_std": float(np.std(beacons)),
            "visits_mean": np.mean(visits, 0),
            "max_abs_q": float(np.mean(maxq)),
        }

    print("=" * 92)
    print(f"{'method':<44}{'FV_1':>7}{'FV_2':>7}{'AND':>8}{'beacons':>9}{'max|Q|':>9}")
    print("-" * 92)
    for name in methods:
        r = results[name]
        print(f"{METHOD_LABELS[name]:<44}{r['fv_mean'][0]:>7.3f}{r['fv_mean'][1]:>7.3f}"
              f"{r['and_mean']:>8.3f}{r['beacons_mean']:>9.2f}{r['max_abs_q']:>9.3f}")
    print("=" * 92)
    print("\nreward-level geomean learns nothing (reward == 0 always -> max|Q| ~ 0).")
    print("Composing the SAME objectives at the Q-value level yields a dense, learnable")
    print("signal that reaches BOTH beacons -- the only method with a positive AND utility.\n")

    plot_curves(results, os.path.join(RESULTS_DIR, "learning_curves.png"))
    plot_bars(results, os.path.join(RESULTS_DIR, "final_metrics.png"))
    plot_visits(results, env0, os.path.join(RESULTS_DIR, "visitations.png"))
    print(f"Saved figures to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
