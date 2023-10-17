from __future__ import annotations
from typing import Optional, List

import numpy as np

from examples.bandits.lib import BanditAgent, Observation, Action


class UcbAgent(BanditAgent):
    """An Upper-Confidence-Bound `Agent` for a `BanditEnvironment`.

    Choose the arm with highest (empirical return + \sqrt(ln(t) / count)).
    """

    def __init__(self):
        """Create an `UcbAgent`."""
        self.last_arm: Optional[int] = None
        self.n_arms: Optional[int] = None

    def learn(self, observation: Observation):
        (n_arms, last_reward) = observation

        if last_reward is None:
            assert self.n_arms is None
            assert self.last_arm is None
            self.n_arms = n_arms
            self.empirical_means: List[float] = [
                0.0 for _ in range(self.n_arms)
            ]
            self.counts: List[int] = [0 for _ in range(self.n_arms)]
            return

        assert self.n_arms is not None
        assert self.last_arm is not None
        self.counts[self.last_arm] += 1
        self.empirical_means[self.last_arm] += (
            last_reward - self.empirical_means[self.last_arm]
        ) / self.counts[self.last_arm]

    def plan(self) -> Action:
        self.last_arm = self._plan()
        return self.last_arm

    def _plan(self) -> Action:
        # pull all arms at least once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # choose an arm greedily
        timestep = sum(self.counts)
        scores = [
            mean + np.sqrt(np.log(timestep) / count)
            for (mean, count) in zip(self.empirical_means, self.counts)
        ]
        return np.argmax(scores)
