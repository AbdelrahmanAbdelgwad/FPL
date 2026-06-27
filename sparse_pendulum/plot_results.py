"""Plot the sparse-Pendulum results written by ``run_sparsity_experiment.py``.

Reads ``results/*.json`` and produces, per ``band``, learning curves of the
*true* objectives (fraction-upright and the intended AND) for each arm, plus a
sample-efficiency summary across bands.

    python experiments/sparse_pendulum/plot_results.py
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ARM_LABELS = {
    "qlevel": "Q-level geomean (BPG)",
    "reward": "reward-level geomean",
    "reward_slack": "reward-level geomean (slack 0.1)",
}
ARM_COLORS = {"qlevel": "#2ca02c", "reward": "#d62728", "reward_slack": "#ff7f0e"}


def load():
    runs = defaultdict(list)  # (band, arm) -> list of curves
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        d = json.load(open(path))
        if d.get("curve"):
            runs[(d["band"], d["arm"])].append(d["curve"])
    return runs


def mean_curve(curves, key):
    steps = sorted({c["steps"] for curve in curves for c in curve})
    series = []
    for s in steps:
        vals = [c[key] for curve in curves for c in curve if c["steps"] == s]
        series.append(np.mean(vals))
    return np.array(steps), np.array(series)


def main():
    runs = load()
    if not runs:
        print(f"No result curves found in {RESULTS_DIR}. Run run_sparsity_experiment.py first.")
        return

    bands = sorted({band for (band, _arm) in runs})
    arms = sorted({arm for (_band, arm) in runs}, key=lambda a: list(ARM_LABELS).index(a))

    # --- per-band learning curves (fraction upright + AND) ---
    fig, axes = plt.subplots(2, len(bands), figsize=(4.6 * len(bands), 7), squeeze=False)
    for j, band in enumerate(bands):
        for arm in arms:
            if (band, arm) not in runs:
                continue
            curves = runs[(band, arm)]
            xs, up = mean_curve(curves, "angle_frac")
            _, andv = mean_curve(curves, "geomean")
            axes[0][j].plot(xs, up, color=ARM_COLORS[arm], lw=2, label=ARM_LABELS[arm])
            axes[1][j].plot(xs, andv, color=ARM_COLORS[arm], lw=2, label=ARM_LABELS[arm])
        axes[0][j].set_title(f"band = {band}  (smaller = sparser)")
        axes[0][j].set_ylabel("fraction of time upright")
        axes[1][j].set_ylabel("intended AND  geomean(upright, actuation)")
        axes[1][j].set_xlabel("environment steps")
        for ax in (axes[0][j], axes[1][j]):
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle("Sparse-objective Pendulum: reward-level vs Q-level geomean composition", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out1 = os.path.join(RESULTS_DIR, "learning_curves.png")
    fig.savefig(out1, dpi=130)
    plt.close(fig)

    # --- per-seed outcomes (honest): every seed's best upright, by band & arm ---
    # Plotting every seed (not a mean line across uneven seed counts) avoids
    # reading a "trend" through single-seed bands. The story is the bimodal
    # solve/fail split, not a smooth average.
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(bands)), 5))
    xticks, xlabels = [], []
    for j, band in enumerate(bands):
        for k, arm in enumerate(arms):
            if (band, arm) not in runs:
                continue
            finals = [max(e["angle_frac"] for e in curve) for curve in runs[(band, arm)]]
            x = j + (k - (len(arms) - 1) / 2) * 0.28
            ax.scatter(x + rng.uniform(-0.05, 0.05, len(finals)), finals,
                       color=ARM_COLORS[arm], s=70, zorder=3,
                       label=ARM_LABELS[arm] if j == 0 else None)
            ax.scatter([x], [np.mean(finals)], color="black", marker="_", s=600, zorder=4)
            ax.annotate(f"n={len(finals)}", (x, -0.06), ha="center", fontsize=7, annotation_clip=False)
        xticks.append(j)
        xlabels.append(f"band {band}\n(±{band*180:.0f}°)")
    ax.axhline(0.5, ls="--", color="0.6", lw=1, label="'solved' threshold")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(-0.02, 1.0)
    ax.set_ylabel("best fraction of time upright (per seed)")
    ax.set_title("Per-seed outcomes (— = mean). Read the split, not a trend across bands.")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    out2 = os.path.join(RESULTS_DIR, "per_seed_outcomes.png")
    fig.savefig(out2, dpi=130)
    plt.close(fig)

    print(f"Saved {out1}\n      {out2}")


if __name__ == "__main__":
    main()
