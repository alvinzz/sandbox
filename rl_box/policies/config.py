from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from math import gamma

import torch


class Config(ABC):
    """Base class for configurations."""

    def verify(self):
        """Raise an error for invalid configuration values."""
        for _, value in self.__dict__():
            if isinstance(value, Config):
                value.verify()


@dataclass
class ActorCriticConfig(Config):
    actor: ActorConfig
    critic: CriticConfig
    gamma: float
    """Time-decay parameter."""
    lambda_: float
    """Trace-decay parameter (for value function estimation). 0 = TD(0), 1 = Monte Carlo."""
    actor_coef: float
    """Relative weight of policy gradient loss."""
    critic_coef: float
    """Relative weight of value estimation loss."""

    def verify(self):
        super().verify()

        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError("`gamma` must be in the range [0.0, 1.0].")

        if self.lambda_ < 0.0 or self.lambda_ > 1.0:
            raise ValueError("`lambda_` must be in the range [0.0, 1.0].")

        if self.actor_coef < 0.0:
            raise ValueError("`actor_coef` must be nonnegative.")

        if self.critic_coef < 0.0:
            raise ValueError("`critic_coef` must be nonnegative.")


@dataclass
class ActorConfig(Config):
    distribution: ActionDistributionConfig
    algo: AlgoConfig
    """Algorithm-specific configuration."""
    normalize_advantages: bool
    """Flag to normalize episodic advantages to mean 0 and stddev 1."""


class ActionDistributionConfig(Config):
    """Marker class for Action Distribution configurations."""

    pass


@dataclass
class CategoricalDistributionConfig(ActionDistributionConfig):
    num_classes: int

    def verify(self):
        super().verify(self)
        if self.num_classes < 1:
            raise ValueError("Number of classes must be positive.")


class AlgoConfig(Config):
    """Marker class for Policy Gradient algorithm configurations."""

    pass


@dataclass
class PpoConfig(AlgoConfig):
    """Configuration for PPO (Proximal Policy Optimization, Schulman et. al., 2017)."""

    clip_coef: float
    """Prevent the ratio of action probabilities from changing more than 1 +/- clip_coef from the data collection policy."""


@dataclass
class CriticConfig(Config):
    huber_loss_coef: float
    """Coefficient for the Huber loss."""

    def get_huber_loss_fn(self):
        return torch.nn.HuberLoss(
            reduction="none", delta=self.huber_loss_coef
        ).forward

    def verify(self):
        super().verify()

        if self.huber_loss_coef < 0.0:
            raise ValueError("`huber_loss_coef` must be nonnegative.")
