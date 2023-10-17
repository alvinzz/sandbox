from __future__ import annotations
from typing import Any, Optional, List, Tuple

import numpy as np

from examples.bandits.lib import BanditAgent, Observation, Action


class BootstrapThompsonAgent(BanditAgent):
    """Thompson Sampling, using the "bootstrap" to sample distributions.

    Gaussian noise is added to the samples, as in [1]
    (and stochastic gradient Langevin dynamics).

    [1] Vaswani et. al., 2018. "New Insights into Bootstrapping for Bandits".
    (Section C.2.1)
    """

    def __init__(self):
        """Create an `BootstrapThompsonAgent`."""
        self.last_arm: Optional[int] = None
        self.n_arms: Optional[int] = None

    def learn(self, observation: Observation):
        (n_arms, last_reward) = observation

        if last_reward is None:
            assert self.n_arms is None
            assert self.last_arm is None
            self.n_arms = n_arms
            self.datasets: Tuple[List[float]] = tuple(
                [] for _ in range(self.n_arms)
            )
            return

        assert self.n_arms is not None
        assert self.last_arm is not None
        self.datasets[self.last_arm].append(last_reward)

    def plan(self) -> Action:
        self.last_arm = self._plan()
        return self.last_arm

    def _plan(self) -> Action:
        # choose an arm greedily
        bootstrapped_means = []

        # pull all arms once
        for arm in range(self.n_arms):
            if len(self.datasets[arm]) == 0:
                return arm

        for arm in range(self.n_arms):
            # add noise to each sample
            noisy_dataset = np.random.normal(
                self.datasets[arm],
                1.0,
            )
            bootstrapped_dataset = np.random.choice(
                noisy_dataset, size=len(self.datasets[arm])
            )
            bootstrapped_means.append(np.mean(bootstrapped_dataset))

        return np.argmax(bootstrapped_means)
