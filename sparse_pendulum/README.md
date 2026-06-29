# Sparse-objective Pendulum: reward-level vs Q-level composition

Compares composing two competing objectives at the **reward level** (scalarize,
then DDPG) against the **Q-value level** (per-objective fulfillment-Q composed in
the actor loss, as in BPG), as one objective is made progressively sparser.

## Environment and objectives

Two competing fulfillments in `[0, 1]` on the classic pendulum:

| objective | signal | type |
|---|---|---|
| `angle` | `1` iff the pole is within `band` of upright, else `0` | binary / sparse |
| `actuation` | `1 - (torque / max_torque)^2` | dense |

`band` is the sparsity knob — the half-width of the "counts as upright" cone, as a
fraction of π:

| band | upright cone |
|---|---|
| 0.10 | ±18° |
| 0.05 | ±9° |
| 0.02 | ±3.6° |

Both objectives are composed with the **geometric mean** (`p = 0`). The experiment
varies only **where** that composition is applied.

## Experiments

| file | what it does |
|---|---|
| `sparse_pendulum.py` | fulfillments, `RewardLevelWrapper`, geomean, and the shared evaluation. Building blocks; not run directly. |
| `run_sparsity_experiment.py` | trains the arms, sweeps `band`; writes `results/*.json`. |
| `mechanism.py` | rolls out a random policy and measures the learning signal of each composition (no training); writes `results/mechanism.png`. |
| `plot_results.py` | writes `learning_curves.png` and `per_seed_outcomes.png` from the JSONs. |

### Arms (`run_sparsity_experiment.py`)

Identical except for where the geomean composes the two objectives:

| arm | composition | code path |
|---|---|---|
| `qlevel` | geomean of the two Q-values in the actor loss | `CMORL(...)`, `p_objectives = 0` |
| `reward` | env returns `geomean(angle, actuation)` as the scalar reward | `RewardLevelWrapper`, `cmorl = None` |
| `reward_slack` | as `reward`, geomean with `slack = 0.1` | `RewardLevelWrapper(slack=0.1)` |
| `qlevel_linear` | compose the Q-values linearly (`p = 1`) | `CMORL(...)`, `p_objectives = 1` |

Every arm is evaluated identically — fraction of time upright, mean actuation, and
their geomean — independent of how it was trained.

### Mechanism probe (`mechanism.py`)

Under a uniform-random policy, measures the per-transition learning signal of each
composition:

- reward-level: `s_r = geomean(angle(t), actuation(t))`
- Q-level proxy: `s_q = geomean(FV_angle(t), FV_actuation(t))`, where
  `FV_k(t) = (1-γ)·Σ_{k≥t} γ^{k-t} r_k` is the normalized discounted return-to-go
  of objective `k`.

Reports, vs `band`, the fraction of transitions with a non-zero signal and the
Q/reward density ratio.

## Results

### `band = 0.05` (±9°), 6 seeds

| | per-seed best fraction upright (seed 0–5) | reach >0.5 |
|---|---|---|
| `qlevel` | 0.93, 0.88, 0.94, 0.94, 0.13, 0.89 | 5 / 6 |
| `reward` | 0.06, 0.20, 0.94, 0.94, 0.09, 0.94 | 3 / 6 |

![per-seed outcomes at band 0.05](results/band0.05_seeds.png)

### Across bands

`band = 0.1` (±18°): both arms reach ~0.95 (1 seed). `band = 0.02` (±3.6°): both
reach ~0.93, later (1 seed).

![learning curves](results/learning_curves.png)
![per-seed outcomes](results/per_seed_outcomes.png)

### Mechanism (`mechanism.png`)

| band | cone | reward-level signal ≠ 0 | Q-level signal ≠ 0 | Q ÷ reward |
|---|---|---|---|---|
| 0.30 | ±54° | 11.3% | 35.1% | ~3× |
| 0.10 | ±18° | 3.3%  | 16.2% | ~5× |
| 0.05 | ±9°  | 1.4%  | 14.4% | ~10× |
| 0.02 | ±4°  | 0.6%  | 11.1% | ~19× |
| 0.005| ±1°  | 0.1%  | 6.9%  | ~46× |

The reward-level signal equals the fraction of transitions where the binary
objective fires; the Q-level signal (returns) stays non-zero across the states
leading to those events.

![mechanism](results/mechanism.png)

## Setup

The BPG core needs TensorFlow; it also imports `wandb` (auto-disabled by the
runner) and `tensorboard`; the plots need `matplotlib`. All run on CPU.

```bash
conda env create --file environment.yml && conda activate cmorl_env   # or: nix develop --impure
pip install wandb tensorboard           # if not already in the env
```

## Reproduce the committed results

Training is seeded with `TF_DETERMINISTIC_OPS` and op determinism enabled; on the
same setup these regenerate the committed `results/*.json` bit-for-bit. The
committed set comes from two configurations:

```bash
# (1) bands 0.1 and 0.02, 1 seed, 8-epoch budget
python sparse_pendulum/run_sparsity_experiment.py \
    --bands 0.1 0.02 --arms qlevel reward --seeds 1 --epochs 8 \
    --steps-per-epoch 1000 --start-steps 500

# (2) band 0.05, 6 seeds, 6-epoch budget
python sparse_pendulum/run_sparsity_experiment.py \
    --bands 0.05 --arms qlevel reward --seeds 6 --epochs 6 \
    --steps-per-epoch 1000 --start-steps 500

# (3) figures from the JSONs
python sparse_pendulum/plot_results.py

# (4) mechanism figure (deterministic, training-free)
python sparse_pendulum/mechanism.py
```

`band0.05_seeds.png` is a per-seed composite; the figures produced by the scripts
above are `learning_curves.png`, `per_seed_outcomes.png`, and `mechanism.png`.

## A consistent sweep (more seeds, more bands, longer budget)

```bash
python sparse_pendulum/run_sparsity_experiment.py \
    --bands 0.15 0.1 0.07 0.05 0.03 0.02 0.01 \
    --arms qlevel qlevel_linear reward reward_slack --seeds 15 --epochs 150
python sparse_pendulum/plot_results.py
```

## Scope of the committed runs

6 seeds at `band = 0.05`; 1 seed at `bands = 0.1, 0.02`; short CPU training budget
(6–8 epochs ≈ 6k–8k env steps per run). Run the consistent sweep above for results
with more seeds and a longer budget.
