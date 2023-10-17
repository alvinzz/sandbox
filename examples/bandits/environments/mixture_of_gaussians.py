from __future__ import annotations
from typing import Tuple

import numpy as np

from examples.bandits.lib import BanditEnvironment, BanditAgent


class MixtureOfGaussiansArms(BanditEnvironment):
    """Environment for a Mixture-of-Gaussians multi-armed bandit.

    Rewards for each arm are sampled from a mixture of Gaussian distributions.
    The Gaussians have means and variances sampled from the Normal-gamma
    distribution (as in the `GaussianArms` environment), while the mixing
    coefficients are sampled from a Dirichlet distribution.
    """

    def __init__(
        self,
        n_arms: int,
        agent: BanditAgent,
        mu_0: float,
        lambda_0: float,
        alpha_0: float,
        beta_0: float,
        dirichlet_coefs: Tuple[float],
    ):
        """Create a Gaussian multi-armed bandit.

        Args:
            n_arms: Number of arms. Must be at least 2.
            agent: A `BanditAgent` which selects an arm to pull at each time-step.
            mu_0: Expected value of the means of the Gaussians.
            lambda_0: Concentration parameter for the means of the Gaussians.
            alpha_0: Shape parameter for the variances of the Gaussians.
            beta_0: Rate parameter for the variance of the Gaussians.
            dirichlet_coefs: Coefficients of the Dirichlet prior over the mixing coeffifients of the Gaussians.
        """
        super().__init__(n_arms, agent)

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

        self.n_gaussians_per_arm = len(dirichlet_coefs)
        self.dirichlet_coefs = dirichlet_coefs

        taus = tuple(
            tuple(
                np.random.gamma(alpha_0, 1.0 / beta_0)
                for _ in range(self.n_gaussians_per_arm)
            )
            for _ in range(self.n_arms)
        )  # invert beta because np.random.gamma uses scale insread of rate

        self.means = tuple(
            tuple(
                np.random.normal(mu_0, np.sqrt(1.0 / (lambda_0 * tau)))
                for tau in arm_taus
            )
            for arm_taus in taus
        )  # sqrt because np.random.normal uses scale instead of variance

        self.variances = tuple(
            tuple(1.0 / tau for tau in arm_taus) for arm_taus in taus
        )
        self.stddevs = tuple(
            tuple(np.sqrt(variance) for variance in arm_variances)
            for arm_variances in self.variances
        )  # amortize computation of Gaussian scales to save compute on calls to np.random.normal

        self.mixing_coefs = tuple(
            np.random.dirichlet(self.dirichlet_coefs)
            for _ in range(self.n_arms)
        )

    def pull_arm(self, arm: int) -> float:
        mixing_coefs = self.mixing_coefs[arm]
        gaussian_idx = np.random.choice(
            self.n_gaussians_per_arm, p=mixing_coefs
        )
        mean = self.means[arm][gaussian_idx]
        stddev = self.stddevs[arm][gaussian_idx]
        return np.random.normal(mean, stddev)
