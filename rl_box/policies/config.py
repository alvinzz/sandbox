from __future__ import annotations

from dataclasses import dataclass
from abc import ABC

import torch


@dataclass
class ActorCriticConfig:
    actor: ActorConfig
    critic: CriticConfig
    gamma: float
    """Time-decay parameter."""
    lambda_: float
    """Trace-decay parameter (for value function estimation). 0 = TD(0), 1 = Monte Carlo."""


@dataclass
class ActorConfig:
    distribution: ActionDistributionConfig
    algo: AlgoConfig
    """Algorithm-specific configuration."""
    normalize_advantages: bool
    """Flag to normalize episodic advantages to mean 0 and stddev 1."""
    entropy_coef: float
    """The multiplier for the entropy of the action distribution that is added to the reward."""


class ActionDistributionConfig(ABC):
    """Marker class for Action Distribution configurations."""

    pass


@dataclass
class CategoricalDistributionConfig(ActionDistributionConfig):
    num_classes: float


class AlgoConfig(ABC):
    """Marker class for Policy Gradient algorithm configurations."""

    pass


@dataclass
class PpoConfig(AlgoConfig):
    """Configuration for PPO (Proximal Policy Optimization, Schulman et. al., 2017)."""

    clip_coef: float
    """Prevent the ratio of action probabilities from changing more than 1 +/- clip_coef from the data collection policy."""


@dataclass
class CriticConfig:
    huber_loss_coef: float
    """Coefficient for the Huber loss."""

    def get_huber_loss_fn(self):
        return torch.nn.HuberLoss(
            reduction="none", delta=self.huber_loss_coef
        ).forward
