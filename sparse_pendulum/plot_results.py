"""Plot the sparse-Pendulum results written by ``run_sparsity_experiment.py``.

Reads ``results/*.json`` and produces:

* ``learning_curves.png``   mean fraction-upright and intended AND vs steps, per band
* ``per_seed_outcomes.png`` every seed's best result, grouped by band and arm
* ``per_seed_band.png``     per-seed traces + outcome strip for the sparsest
                            well-sampled band

    python sparse_pendulum/plot_results.py
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

# Colorblind-safe categorical assignment (validated: worst adjacent CVD dE 96.7,
# vs 7.3 for the previous red/green pair). Color follows the arm, never its rank.
ARM_LABELS = {
    "qlevel": "Q-level geomean (BPG)",
    "qlevel_linear": "Q-level linear (p=1, ablation)",
    "reward": "reward-level geomean",
    "reward_slack": "reward-level geomean (slack 0.1)",
}
ARM_COLORS = {
    "qlevel": "#2a78d6",         # blue
    "reward": "#eb6834",         # orange
    "qlevel_linear": "#1baf7a",  # aqua
    "reward_slack": "#4a3aa7",   # violet
}

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})

SOLVED = 0.5  # "solved" threshold on fraction of time upright


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


def band_label(band, n=None):
    lab = f"±{band * 180:.0f}°"
    return f"{lab}\nn={n}/arm" if n is not None else lab


def solved_count(curves):
    return sum(max(e["angle_frac"] for e in c) > SOLVED for c in curves)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_learning_curves(runs, bands, arms, path):
    fig, axes = plt.subplots(2, len(bands), figsize=(4.4 * len(bands), 7.2),
                             squeeze=False, layout="constrained", sharey="row")
    for j, band in enumerate(bands):
        n = max(len(runs[(band, a)]) for a in arms if (band, a) in runs)
        for arm in arms:
            if (band, arm) not in runs:
                continue
            curves = runs[(band, arm)]
            xs, up = mean_curve(curves, "angle_frac")
            _, andv = mean_curve(curves, "geomean")
            axes[0][j].plot(xs, up, color=ARM_COLORS[arm], lw=2, label=ARM_LABELS[arm])
            axes[1][j].plot(xs, andv, color=ARM_COLORS[arm], lw=2)
        axes[0][j].set_title(f"band {band}  ({band_label(band).splitlines()[0]}, n={n}/arm)")
        axes[1][j].set_xlabel("environment steps")
        for ax in (axes[0][j], axes[1][j]):
            ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("mean fraction of time upright")
    axes[1][0].set_ylabel("mean AND  geomean(upright, actuation)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels))
    fig.savefig(path)
    plt.close(fig)


def plot_per_seed_outcomes(runs, bands, arms, path):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(max(7.5, 2.6 * len(bands)), 5.2), layout="constrained")
    xticks, xlabels = [], []
    for j, band in enumerate(bands):
        n = max(len(runs[(band, a)]) for a in arms if (band, a) in runs)
        for k, arm in enumerate(arms):
            if (band, arm) not in runs:
                continue
            finals = [max(e["angle_frac"] for e in c) for c in runs[(band, arm)]]
            x = j + (k - (len(arms) - 1) / 2) * 0.32
            ax.scatter(x + rng.uniform(-0.06, 0.06, len(finals)), finals,
                       color=ARM_COLORS[arm], s=55, zorder=3,
                       edgecolors="white", linewidths=0.7,
                       label=ARM_LABELS[arm] if j == 0 else None)
            ax.hlines(np.mean(finals), x - 0.14, x + 0.14, color="0.15", lw=2.2, zorder=4)
            ax.annotate(f"{solved_count(runs[(band, arm)])}/{len(finals)}",
                        (x, 1.03), ha="center", fontsize=9, color="0.25",
                        annotation_clip=False)
        xticks.append(j)
        xlabels.append(f"band {band}\n{band_label(band, n)}")
    ax.axhline(SOLVED, ls="--", color="0.6", lw=1)
    ax.text(len(bands) - 0.52, SOLVED + 0.015, "solved", fontsize=9, color="0.45",
            ha="right", va="bottom")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlim(-0.6, len(bands) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("best fraction of time upright (per seed)")
    ax.set_title("Per-seed outcomes by sparsity band   (— = mean, counts = solved/seeds)",
                 pad=22)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def plot_per_seed_band(runs, arms, path):
    """Per-seed traces (thin) + mean (thick) + outcome strip for the sparsest of
    the most-sampled bands (where any effect is largest)."""
    counts = {}
    for (band, _arm), curves in runs.items():
        counts[band] = counts.get(band, 0) + len(curves)
    most = max(counts.values())
    band = min(b for b in counts if counts[b] == most)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), layout="constrained", width_ratios=[1.7, 1])
    rng = np.random.default_rng(0)
    present = [a for a in arms if (band, a) in runs]
    for i, arm in enumerate(present):
        curves = runs[(band, arm)]
        for c in curves:
            ax1.plot([e["steps"] for e in c], [e["angle_frac"] for e in c],
                     color=ARM_COLORS[arm], alpha=0.16, lw=1)
        x, m = mean_curve(curves, "angle_frac")
        ax1.plot(x, m, color=ARM_COLORS[arm], lw=2.6,
                 label=f"{ARM_LABELS[arm]}  ·  {solved_count(curves)}/{len(curves)} solve")
        finals = [max(e["angle_frac"] for e in c) for c in curves]
        ax2.scatter(np.full(len(finals), i) + rng.uniform(-0.07, 0.07, len(finals)),
                    finals, color=ARM_COLORS[arm], s=55, zorder=3,
                    edgecolors="white", linewidths=0.7)
        ax2.hlines(np.mean(finals), i - 0.18, i + 0.18, color="0.15", lw=2.2, zorder=4)
    for ax in (ax1, ax2):
        ax.axhline(SOLVED, ls="--", color="0.6", lw=1)
        ax.set_ylim(0, 1)
    ax1.set_xlabel("environment steps")
    ax1.set_ylabel("fraction of time upright")
    ax1.set_title(f"band {band} ({band_label(band).splitlines()[0]}): every seed (thin) + mean (thick)")
    ax1.legend(fontsize=9, loc="upper left")
    ax2.set_xticks(range(len(present)))
    ax2.set_xticklabels([ARM_LABELS[a].split(" (")[0] for a in present], fontsize=9)
    ax2.set_xlim(-0.6, len(present) - 0.4)
    ax2.set_ylabel("best fraction upright (per seed)")
    ax2.set_title(f"band {band}: per-seed outcomes (— = mean)")
    ax2.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)
    return band


def main():
    runs = load()
    if not runs:
        print(f"No result curves found in {RESULTS_DIR}. Run run_sparsity_experiment.py first.")
        return

    # sparsest band first, so reading order matches the story
    bands = sorted({band for (band, _arm) in runs})
    arms = [a for a in ARM_LABELS if any((b, a) in runs for b in bands)]

    out1 = os.path.join(RESULTS_DIR, "learning_curves.png")
    out2 = os.path.join(RESULTS_DIR, "per_seed_outcomes.png")
    out3 = os.path.join(RESULTS_DIR, "per_seed_band.png")
    plot_learning_curves(runs, bands, arms, out1)
    plot_per_seed_outcomes(runs, bands, arms, out2)
    plot_per_seed_band(runs, arms, out3)
    print(f"Saved {out1}\n      {out2}\n      {out3}")


if __name__ == "__main__":
    main()
