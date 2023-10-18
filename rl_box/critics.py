"""Critics for RL.
"""

from abc import ABC, abstractmethod

import torch


class ReturnEstimator(ABC):
    """Abstract class for estimators of expected returns (value functions)."""

    def __init__(self, observation_dim: torch.Tensor):
        self.observation_dim = observation_dim

    @property
    def o(self):
        return self.observation_dim

    @abstractmethod
    def value(self, observation: torch.Tensor) -> torch.Tensor:
        """Get the estimated value (expected return) for a given observation.

        Args:
            observation: Tensor with shape (*, O).

        Returns:
            value: Floating-point Tensor with shape (*).
        """
        raise NotImplementedError


class ZeroReturnEstimator(ReturnEstimator):
    """A value function that always returns 0."""

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.zeros(observation.shape[:-1])
