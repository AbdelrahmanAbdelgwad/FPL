# Sparse-objective Pendulum: reward-level vs Q-level composition

This experiment tests the original hypothesis — *does composing objectives at the
**Q-value level** help when rewards are sparse?* — **in the regime where it can
actually matter**: a gradient-based actor (BPG/DDPG) optimizing **simultaneous,
competing** objectives, one of which is **sparse / binary**.

## Why here and not a gridworld

A companion study (branch `claude/gallant-sagan-5yfwqm`, the `TwoBeaconGridworld`)
showed that a *tabular* gridworld is the wrong testbed: there the policy learns by
`argmax` (no gradient to smooth), and the only place reward-level composition is
truly degenerate is a task whose optimum is a non-stationary orbit a memoryless
table can't represent. The benefit of Q-level composition is about giving a
**gradient-based learner a smooth, dense signal** out of sparse per-objective
rewards — so the right testbed is continuous control with an actor network and
**ongoing** (not sequential) competing objectives. That is exactly Pendulum.

## Setup

Two competing fulfillments on the pendulum:

| objective | signal | type |
|---|---|---|
| `angle` | `1` iff the pole is within `band` of upright, else `0` | **binary / sparse** |
| `actuation` | `1 - (torque/max)^2` | dense |

`band` is the **sparsity knob**: large → upright fires often (dense-ish); small →
it fires only very near the top (very sparse, "0 on the lower half").

Both objectives are composed with the **geometric mean** (logical AND, `p = 0`).
The arms differ **only in where that composition happens**:

| arm | how it composes | code path |
|---|---|---|
| `qlevel` | geomean of the two **Q-values** in the actor loss (BPG) | `CMORL(...)`, `p_objectives = 0` |
| `reward` | env returns `geomean(angle, actuation)` as a **scalar reward**, plain DDPG | `RewardLevelWrapper`, `cmorl = None` |
| `reward_slack` | same, but geomean with `slack = 0.1` (Bassel's "never output exactly 0") | `RewardLevelWrapper(slack=0.1)` |

The prediction: as `band → 0`, the **reward-level** scalar is `0` almost
everywhere (no gradient toward upright until the agent stumbles onto the top),
while the **Q-level** arm keeps a dense `angle` fulfillment-Q that bootstraps from
the rare upright visits and composes into a usable actor gradient. So Q-level
should degrade much more gracefully with sparsity.

## Run

```bash
# quick local check (CPU, short — sanity, not a full result):
python sparse_pendulum/run_sparsity_experiment.py \
    --bands 0.1 --arms qlevel reward --seeds 1 --epochs 15 --steps-per-epoch 1000 --start-steps 600

# fuller study (run in the conda/nix env, ideally on GPU):
python sparse_pendulum/run_sparsity_experiment.py \
    --bands 0.5 0.25 0.1 0.05 --arms qlevel reward reward_slack --seeds 3 --epochs 150
python sparse_pendulum/plot_results.py
```

`wandb` is set to `disabled` automatically; the BPG core also needs `tensorboard`
installed (it writes `tf.summary` logs). Every arm is scored by the **same**
evaluation — fraction of time upright, mean actuation, and their geomean — so the
comparison is on the true objectives regardless of how each arm was trained.
Results are written to `results/*.json`; `plot_results.py` makes the figures.

## Results (preliminary)

Verified end-to-end under TensorFlow-CPU. These are **short, CPU-budget** runs
(6 epochs ≈ 6k env steps each) — directional, not publication-grade — committed
under `results/` with the figures (`learning_curves.png`, `per_seed_outcomes.png`, `band0.05_seeds.png`):

| band (sparsity) | seeds | Q-level learns | reward-level learns |
|---|---|---|---|
| 0.10 (mild)   | 1 | yes | yes (top is hit often → reward gets signal) |
| **0.05**      | **6** | **5 / 6** | **3 / 6** |
| 0.02 (extreme)| 1 | yes (late) | yes (late, noisy) |

"learns" = greedy policy reaches >50% time-upright within the 6-epoch budget.

**Reading it honestly:**

- The hypothesis holds *directionally* in the right regime: at `band = 0.05`
  the Q-level arm is **more sample-efficient / reliable** (5/6 vs 3/6 within the
  same budget), and its mean learning curve is clearly above the reward-level
  arm's (see `learning_curves.png`).
- But it is a **reliability/efficiency edge, not a clean "impossible" gap.** On
  the pendulum the top is visited often enough during exploration that the
  reward-level geomean usually gets *some* upright samples and frequently
  recovers — so reward-level still learns about half the time. This matches the
  paper's framing (a sample-efficiency improvement) rather than the degenerate
  "0 everywhere" extreme.
- 5/6 vs 3/6 on 6 seeds is **suggestive, not significant.** A defensible claim
  needs the "fuller study" command (more seeds, the band sweep, and the paper's
  long training budget) run in your conda/nix env — ideally on GPU.

## Caveats

`wandb` is auto-disabled; the BPG core also imports `tensorboard` (for
`tf.summary`) and the plots need `matplotlib` — all present in the repo's
conda/nix env. Background training was unstable in the sandbox used to verify
this (processes were signal-killed mid-run), so the committed numbers came from
short foreground runs; this is an environment quirk, not a code issue.
