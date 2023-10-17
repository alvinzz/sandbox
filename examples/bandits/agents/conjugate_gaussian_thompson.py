from __future__ import annotations
from typing import Optional, List

import numpy as np

from examples.bandits.lib import BanditAgent, Observation, Action


class ConjugateGaussianThompsonAgent(BanditAgent):
    """Thompson sampling, assuming a conjugate Gaussian prior (Normal-gamma).

    Model the returns from each arm as a Gaussian distribution with parameters
    drawn from a conjugate prior distribution (the Normal-gamma). At each step,
    sample a Gaussian distribution for each arm and select the arm with
    highest expected value.
    """

    def __init__(
        self,
        mu_0: float = 0.0,
        lambda_0: float = 1.0,
        alpha_0: float = 2.0,
        beta_0: float = 2.0,
    ):
        """Create an `ConjugateGaussianThompsonAgent`.

        Arguments are the prior belief over parameters of the Normal-gamma
        distribution from which parameters of the Gaussians are drawn.

        Args:
            mu_0: Empirical mean (including prior pseudo-observations).
            lambda_0: Number of observations of the mean (including the prior).
            alpha_0: 0.5 times the number of observations of the variance (including the prior).
            beta_0: 0.5 times the empirical sum of squared deviations (inluding prior psuedo-observations).
        """
        if lambda_0 < 0.0:
            raise ValueError("`lambda_0` must be positive.")

        if alpha_0 < 0.0:
            raise ValueError("`alpha_0` must be positive.")

        if beta_0 < 0.0:
            raise ValueError("`beta_0` must be positive.")

        self.mu_0 = mu_0
        self.lambda_0 = lambda_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        self.last_arm: Optional[int] = None
        self.n_arms: Optional[int] = None

    def learn(self, observation: Observation):
        (n_arms, last_reward) = observation

        if last_reward is None:
            assert self.n_arms is None
            assert self.last_arm is None
            self.n_arms = n_arms
            self.empirical_means: List[float] = [
                self.mu_0 for _ in range(self.n_arms)
            ]
            self.mean_counts: List[float] = [
                self.lambda_0 for _ in range(self.n_arms)
            ]
            self.variance_counts: List[float] = [
                2.0 * self.alpha_0 for _ in range(self.n_arms)
            ]
            self.sum_sq_deviations: List[float] = [
                2.0 * self.beta_0 for _ in range(self.n_arms)
            ]
            return

        assert self.n_arms is not None
        assert self.last_arm is not None
        old_mean = self.empirical_means[self.last_arm]
        self.mean_counts[self.last_arm] += 1.0
        self.variance_counts[self.last_arm] += 1.0
        self.empirical_means[self.last_arm] += (
            last_reward - self.empirical_means[self.last_arm]
        ) / self.mean_counts[self.last_arm]
        self.sum_sq_deviations[self.last_arm] += (last_reward - old_mean) * (
            last_reward - self.empirical_means[self.last_arm]
        )

    def plan(self) -> Action:
        self.last_arm = self._plan()
        return self.last_arm

    def _plan(self) -> Action:
        # sample a Gaussian for each arm
        mu_0s = self.empirical_means
        lambda_0s = self.mean_counts
        alpha_0s = [count / 2.0 for count in self.variance_counts]
        beta_0s = [
            sum_sq_deviations / 2.0
            for sum_sq_deviations in self.sum_sq_deviations
        ]

        taus = tuple(
            np.random.gamma(alpha_0, 1.0 / beta_0)
            for (alpha_0, beta_0) in zip(alpha_0s, beta_0s)
        )  # invert beta because np.random.gamma uses scale insread of rate

        means = tuple(
            np.random.normal(mu_0, np.sqrt(1.0 / (lambda_0 * tau)))
            for (mu_0, lambda_0, tau) in zip(mu_0s, lambda_0s, taus)
        )  # sqrt because np.random.normal uses scale instead of variance

        # Thompson: choose the sampled Gaussian with the highest expected return
        return np.argmax(means)
