from __future__ import annotations
from typing import Optional, List

import numpy as np

from examples.bandits.lib import BanditAgent, Observation, Action


class EpsilonGreedyAgent(BanditAgent):
    """An episilon-greedy `Agent` for a `BanditEnvironment`.

    Choose a random arm with probability `epsilon`, else choose the arm with highest empirical return.
    """

    def __init__(self, epsilon: float, decay: float = 1.0):
        """Create an `EpsilonGreedyAgent`.

        Args:
            epsilon: Chance of choosing randomly (instead of greedily).
            decay: Rate of decay of `epsilon`.
        """
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("`epsilon` must be between 0.0 and 1.0.")

        self.epsilon_0 = epsilon
        self.epsilon = epsilon
        self.decay = decay

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

        self.epsilon *= self.decay

    def plan(self) -> Action:
        self.last_arm = self._plan()
        return self.last_arm

    def _plan(self) -> Action:
        # pull all arms at least once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # with probability `epsilon`, pull a random arm
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)

        # else choose an arm greedily
        return np.argmax(self.empirical_means)
