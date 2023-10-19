from __future__ import annotations
from cmath import log

from typing import Callable, Tuple
import warnings

import torch

from rl_box.policies.policy import Policy, ActionDistribution
from rl_box.policies.distributions import CategoricalDistribution
from rl_box.policies.config import (
    ActorCriticConfig,
    CategoricalDistributionConfig,
    PpoConfig,
)
from rl_box.traces import lambda_returns_and_advantages
from learner import GradientLearner


Observations = torch.Tensor  # [B, T + 1, O]
Values = torch.Tensor  # [B, T + 1, O]
ActionDistributionParams = torch.Tensor  # [B, T, D]
Actions = torch.Tensor  # [B, T, A]
Rewards = torch.Tensor  # [B, T]
Dones = torch.Tensor  # [B, T]
EpisodicData = Tuple[
    Observations, Values, ActionDistributionParams, Actions, Rewards, Dones
]


class ActorCritic(Policy):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        params: ActorCriticConfig,
    ):
        super().__init__(
            observation_dim,
            action_dim,
        )

        self.network = network
        self.optimizer = optimizer
        self.params = params

        if isinstance(
            self.params.actor.distribution, CategoricalDistributionConfig
        ):
            action_distribution = CategoricalDistribution(
                self.params.actor.distribution
            )
        else:
            raise NotImplementedError

        self.action_distribution: ActionDistribution = action_distribution

        self.loss_fn = self.get_loss_fn()

        self.learner = GradientLearner(
            self.network, self.optimizer, self.loss_fn
        )

    def forward(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return action distribution parameters and value function estimates for a sequence of observations.

        Args:
            observations: Tensor of shape [*, T, O].

        Returns:
            action_distribution_params: Tensor of shape [*, T, D].

        """
        output = self.network(observations)  # [*, T, D + 1]
        action_distribution_params = output[..., :-1]  # [*, T, D]
        value = output[..., -1]  # [*, T]
        return action_distribution_params, value

    def action_distribution_params(
        self, observations: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(observations)[0]

    def get_loss_fn(self) -> Callable[[EpisodicData], torch.Tensor]:
        if isinstance(self.params.actor.algo, PpoConfig):
            return self.get_ppo_loss_fn()
        else:
            raise NotImplementedError

    def get_ppo_loss_fn(self) -> Callable[[EpisodicData], torch.Tensor]:
        algo_params: PpoConfig = self.params.actor.algo

        critic_loss_fn = self.params.critic.get_huber_loss_fn()

        def loss_fn(data: EpisodicData) -> torch.Tensor:
            data = map(lambda tensor: tensor.detach(), data)
            (
                observations,  # [B, T + 1, O]
                old_values,  # [B, T + 1]
                old_action_distribution_params,  # [B, T + 1, D]
                actions,  # [B, T, A]
                rewards,  # [B, T]
                dones,  # [B, T]
            ) = data

            # past episode statistics
            old_log_probs = self.action_distribution.log_prob(
                old_action_distribution_params, actions
            )  # [B, T]
            lambda_returns, lambda_advantages = lambda_returns_and_advantages(
                rewards,
                dones,
                old_values,
                self.params.gamma,
                self.params.lambda_,
            )  # [B, T], [B, T]

            # current policy statistics
            action_distribution_params, values = self.forward(
                observations[..., :-1, :]
            )  # [B, T, D], [B, T]
            log_probs = self.action_distribution.log_prob(
                action_distribution_params, actions
            )  # [B, T]

            # mask invalid entries
            T = dones.shape[1]
            num_invalid = torch.sum(dones, dim=1) - 1  # [B], int
            valid_mask = torch.arange(T).unsqueeze(0) < (
                T - num_invalid.unsqueeze(1)
            )  # [B, T], bool
            lambda_advantages = lambda_advantages[valid_mask]  # [V]
            log_probs = log_probs[valid_mask]  # [V]
            old_log_probs = old_log_probs[valid_mask]  # [V]
            lambda_returns = lambda_returns[valid_mask]  # [V]
            values = values[valid_mask]  # [V]

            if self.params.actor.normalize_advantages:
                if lambda_advantages.size() == 1:
                    warnings.warn(
                        "Only one observation, skipping advantage normalization!"
                    )
                else:
                    lambda_advantages = (
                        lambda_advantages - lambda_advantages.mean()
                    ) / (lambda_advantages.std() + 1e-8)

            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs)  # [V]
            clamp_ratio = torch.clamp(
                ratio,
                1.0 - algo_params.clip_coef,
                1.0 + algo_params.clip_coef,
            )  # [V]
            actor_loss = torch.min(
                lambda_advantages * ratio,
                lambda_advantages * clamp_ratio,
            )  # [V]

            # value estimation loss
            critic_loss = critic_loss_fn(lambda_returns, values)  # [V]

            loss = (
                self.params.critic_coef * critic_loss
                - self.params.actor_coef * actor_loss
            ).mean()

            return loss

        return loss_fn

    def update(self, data: EpisodicData):
        self.learner.update((data,))
