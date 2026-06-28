"""Train the reward-level vs Q-level arms on sparse-objective Pendulum and record
a common evaluation curve, swept over the sparsity knob ``band``.

    # quick local check (short, CPU-friendly):
    python experiments/sparse_pendulum/run_sparsity_experiment.py \
        --bands 0.3 0.1 --arms qlevel reward --seeds 1 --epochs 30 --steps-per-epoch 1000

    # fuller study (run in your conda/nix env, ideally on GPU):
    python experiments/sparse_pendulum/run_sparsity_experiment.py \
        --bands 0.5 0.25 0.1 0.05 --arms qlevel reward reward_slack --seeds 3 --epochs 150

Each arm differs *only* in where the geomean(angle, actuation) composition happens:
  qlevel       -> CMORL vector reward, composed at the Q-value level (p_objectives=0)
  reward       -> env returns geomean(...) as scalar reward, plain DDPG (slack 0)
  reward_slack -> same, but geomean with slack=0.1 ("never output exactly 0")
"""

from __future__ import annotations

import os

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparse_pendulum import (  # noqa: E402
    RewardLevelWrapper,
    evaluate_policy,
    make_cmorl,
)
import Pendulum  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def build_hypers(seed: int, epochs: int, steps_per_epoch: int, start_steps: int = 1000,
                 p_objectives: float = 0.0):
    from cmorl.rl_algs.ddpg.hyperparams import default_hypers
    hp = default_hypers()
    hp.seed = seed
    hp.epochs = epochs
    hp.steps_per_epoch = steps_per_epoch
    hp.max_ep_len = 200
    hp.start_steps = start_steps
    hp.gamma = 0.99
    hp.polyak = 0.9
    hp.pi_lr = 3e-3
    hp.q_lr = 3e-3
    hp.act_noise = 0.05
    hp.qd_power = 0.75
    hp.before_clip = 1.0
    hp.p_batch = 1.0
    hp.p_objectives = p_objectives  # 0 = geomean (AND); 1 = linear, at the Q level
    hp.ac_kwargs = {"actor_hidden_sizes": [32, 32], "critic_hidden_sizes": [400, 300]}
    return hp


def train_arm(arm: str, band: float, seed: int, epochs: int, steps_per_epoch: int,
              start_steps: int = 1000) -> list[dict]:
    """Train one arm; return the evaluation curve (one entry per epoch)."""
    from cmorl.rl_algs.ddpg.ddpg import ddpg

    # qlevel_linear is the ablation: compose the Q-values *linearly* (p=1) instead
    # of with the geomean, to separate "compose at the Q level" from "the AND".
    p_objectives = 1.0 if arm == "qlevel_linear" else 0.0
    hp = build_hypers(seed, epochs, steps_per_epoch, start_steps, p_objectives)
    curve: list[dict] = []

    def on_save(pi_network, _q_network, epoch):
        m = evaluate_policy(pi_network, band)
        m["epoch"] = int(epoch)
        m["steps"] = int(epoch) * steps_per_epoch
        curve.append(m)
        print(f"  [{arm} band={band} seed={seed}] epoch {epoch:>3}  "
              f"upright={m['angle_frac']:.2f} actuation={m['actuation']:.2f} "
              f"AND={m['geomean']:.3f}", flush=True)

    if arm in ("qlevel", "qlevel_linear"):
        env_fn = lambda: Pendulum.PendulumEnv(g=10.0, setpoint=0.0)
        ddpg(env_fn, experiment_name=f"{arm}_b{band}_s{seed}", hp=hp,
             cmorl=make_cmorl(band), on_save=on_save,
             experiment_description=f"Q-level composition (p_objectives={hp.p_objectives})")
    elif arm in ("reward", "reward_slack"):
        slack = 0.1 if arm == "reward_slack" else 0.0
        env_fn = lambda: RewardLevelWrapper(Pendulum.PendulumEnv(g=10.0, setpoint=0.0),
                                            band=band, slack=slack)
        ddpg(env_fn, experiment_name=f"{arm}_b{band}_s{seed}", hp=hp,
             cmorl=None, on_save=on_save,
             experiment_description="Reward-level geomean composition (scalar DDPG)")
    else:
        raise ValueError(arm)
    return curve


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bands", type=float, nargs="+", default=[0.3, 0.1])
    parser.add_argument("--arms", nargs="+", default=["qlevel", "reward"],
                        choices=["qlevel", "qlevel_linear", "reward", "reward_slack"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--start-steps", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"bands={args.bands} arms={args.arms} seeds={args.seeds} "
          f"epochs={args.epochs} steps/epoch={args.steps_per_epoch}\n")

    for band in args.bands:
        for arm in args.arms:
            for seed in range(args.seeds):
                curve = train_arm(arm, band, seed, args.epochs, args.steps_per_epoch,
                                  args.start_steps)
                out = os.path.join(RESULTS_DIR, f"{arm}_band{band}_seed{seed}.json")
                with open(out, "w") as f:
                    json.dump({"arm": arm, "band": band, "seed": seed, "curve": curve}, f, indent=2)
                print(f"  -> saved {out}\n", flush=True)


if __name__ == "__main__":
    main()
