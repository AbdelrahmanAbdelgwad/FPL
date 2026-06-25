"""TwoBeaconGridworld: testing Q-level composition of sparse binary objectives.

See ``README.md`` for the hypothesis and ``run_experiment.py`` for the
reward-level vs Q-level comparison.
"""

from .two_beacon_gridworld import TwoBeaconGridworldEnv  # noqa: F401

__all__ = ["TwoBeaconGridworldEnv"]
