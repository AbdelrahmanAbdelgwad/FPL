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

    # --- sample-efficiency summary: final upright fraction vs band ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm in arms:
        xs, ys = [], []
        for band in bands:
            if (band, arm) not in runs:
                continue
            finals = [curve[-1]["angle_frac"] for curve in runs[(band, arm)]]
            xs.append(band)
            ys.append(np.mean(finals))
        ax.plot(xs, ys, "o-", color=ARM_COLORS[arm], lw=2, label=ARM_LABELS[arm])
    ax.invert_xaxis()  # sparser to the right
    ax.set_xlabel("band  (← denser     sparser →)")
    ax.set_ylabel("final fraction of time upright")
    ax.set_title("Where does reward-level composition break as the objective gets sparse?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out2 = os.path.join(RESULTS_DIR, "sparsity_summary.png")
    fig.savefig(out2, dpi=130)
    plt.close(fig)

    print(f"Saved {out1}\n      {out2}")


if __name__ == "__main__":
    main()
