from __future__ import annotations

import numpy as np

from examples.bandits.lib import BanditEnvironment, BanditAgent


class GaussianArms(BanditEnvironment):
    """Environment for a Gaussian multi-armed bandit.

    Rewards for arm `i` are sampled from a Gaussian distribution with mean`mu_i`
    and variance `1 / tau_i`.

    `tau_i`'s are sampled i.i.d. from Gamma(`alpha_0`, `beta_0`) and `mu_i`s
    from N(`mu_0`, (`\lambda_0` `tau_i`)^(-1)).

    (Gamma mean: `alpha_0` / `beta_0`, Gamma variance: `alpha_0` / `beta_0`^2.)
    """

    def __init__(
        self,
        n_arms: int,
        agent: BanditAgent,
        mu_0: float,
        lambda_0: float,
        alpha_0: float,
        beta_0: float,
    ):
        """Create a Gaussian multi-armed bandit.

        Args:
            n_arms: Number of arms. Must be at least 2.
            agent: A `BanditAgent` which selects an arm to pull at each time-step.
            mu_0: Mean of pseudo-observations.
            lambda_0: Number of pseudo-observations of the mean.
            alpha_0: 0.5 times the number of pseudo-observations of the variance.
            beta_0: 0.5 times the sum of squared deviations of pseudo-observations.
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

        taus = tuple(
            np.random.gamma(alpha_0, 1.0 / beta_0) for _ in range(self.n_arms)
        )  # invert beta because np.random.gamma uses scale insread of rate

        self.means = tuple(
            np.random.normal(mu_0, np.sqrt(1.0 / (lambda_0 * tau)))
            for tau in taus
        )  # sqrt because np.random.normal uses scale instead of variance

        self.variances = tuple(1.0 / tau for tau in taus)
        self.stddevs = tuple(
            np.sqrt(variance) for variance in self.variances
        )  # amortize computation of Gaussian scales to save compute on calls to np.random.normal

    def pull_arm(self, arm: int) -> float:
        mean = self.means[arm]
        stddev = self.stddevs[arm]
        return np.random.normal(mean, stddev)
