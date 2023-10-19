"""Policies for RL.
"""

from abc import ABC, abstractmethod

import torch

from rl_box.policies.distributions import ActionDistribution


class Policy(ABC):
    """Abstract RL policy."""

    def __init__(
        self, observation_dim: int, action_distribution: ActionDistribution
    ):
        self.observation_dim = observation_dim
        self.action_distribution = action_distribution

    @property
    def o(self):
        return self.observation_dim

    @abstractmethod
    def action_distribution_params(
        self, observations: torch.Tensor
    ) -> torch.Tensor:
        """Return action distribution parameters for a sequence of observations.

        Args:
            observations: Tensor with shape [*, T, O].

        Returns:
            action_distribution_params: Tensor with shape [*, T, D].
        """
        raise NotImplementedError
