"""Experiments on multi-armed bandits.

Environments:
    - Bernoulli: Arms return either 0 or 1, with mean \theta. \theta is sampled from Beta(\alpha, \beta).
    - Gaussian: The return from each arm is distributed as N(\mu, \tau^{-1}), where \mu is sampled from N(\mu_0, (\lambda_0 \tau)^{-1}) and \tau is sampled from Gamma(\alpha_0, \beta_0).
    - Mixture-of-Gaussians: As above, but using a mixture of K Gaussians, with weights sampled from Dirichlet(\alpha_1, ..., \alpha_K).

Classical Algorithms:
    - \epsilon-greedy: Choose a random arm with probability \epsilon, else choose the arm with highest empirical return.
    - UCB: Choose the arm with highest (empirical return + an exploration term).
    - conjugate-Gaussian-Thompson: Model the returns from each arm as a Gaussian distribution with parameters drawn from a conjugate prior distribution (the Normal-gamma). At each step, sample a Gaussian distribution for each arm and select the arm with highest expected value.
    - bootstrapped-Thompson: At each step, for each arm, sample a random bootstrapped dataset. Add Gaussian noise with variance \sigma to the samples (as in Langevin dynamics).* Select the arm with the highest expected value on the bootstrapped dataset.

Learning Algorithms:
    - ...

*Vaswani et. al., 2018. "New Insights into Bootstrapping for Bandits". (Section C.2.1)
"""

from __future__ import annotations
from typing import Optional, Tuple
from abc import abstractmethod

from agent import Agent
from environment import Environment


class BanditEnvironment(Environment):
    """A collection of arms with variable rewards.

    Contains exactly one `BanditAgent`. At each time-step, the `BanditAgent`
    selects an arm to pull.

    Contains two observable quantities:
      - the number of arms, and
      - the reward from the last arm which was pulled.

    Args:
        n_arms: Number of arms. Must be at least 2.
        agent: A `BanditAgent` which selects an arm to pull at each time-step.
    """

    def __init__(self, n_arms: int, agent: BanditAgent):
        if n_arms < 2:
            raise ValueError("Must have at least 2 arms.")

        self.n_arms = n_arms
        self.agent = agent

        self.last_reward = None

    def step(self):
        arm = self.agent.act(self)
        self.last_reward = self.pull_arm(arm)

    @abstractmethod
    def pull_arm(self, arm: int) -> float:
        """Pull the an arm and receive a reward."""
        raise NotImplementedError


Observation = Tuple[int, Optional[float]]
"""Observation for a `BanditAgent`. Contains (n_arms, last_reward)."""
Action = int
"""Action of a `BanditAgent`. Must be an integer in [0, n_arms)."""


class BanditAgent(Agent[Observation, Action]):
    """An `Agent` for a `BanditEnvironment` environment."""

    def observe(self, environment: BanditEnvironment) -> Observation:
        return (environment.n_arms, environment.last_reward)
