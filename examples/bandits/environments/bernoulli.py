from __future__ import annotations

import numpy as np

from examples.bandits.lib import BanditEnvironment, BanditAgent


class BernoulliArms(BanditEnvironment):
    """Environment for a Bernoulli multi-armed bandit.

    Arms return either 0 or 1, with mean `theta`. `theta` is sampled from Beta(`alpha`, `beta`).
    """

    def __init__(
        self, n_arms: int, agent: BanditAgent, alpha: float, beta: float
    ):
        """Create a Benoulli multi-armed bandit.

        E[`theta`] = `alpha` / (`alpha` + `beta`).
        `alpha`, `beta` >> 1 -> Samples of `theta` will be concentrated around the mean.

        Args:
            n_arms: Number of arms. Must be at least 2.
            agent: A `BanditAgent` which selects an arm to pull at each time-step.
            alpha: `alpha` parameter of the Beta distribution for the arm means.
            beta: `beta` parameter of the Beta distribution for the arm means.
        """
        super().__init__(n_arms, agent)

        if alpha < 0.0:
            raise ValueError("`alpha` must be positive.")

        if beta < 0.0:
            raise ValueError("`beta` must be positive.")

        self.alpha = alpha
        self.beta = beta
        self.thetas = tuple(
            np.random.beta(alpha, beta) for _ in range(self.n_arms)
        )

    def pull_arm(self, arm: int) -> float:
        if np.random.rand() < self.thetas[arm]:
            return 1.0
        return 0.0
