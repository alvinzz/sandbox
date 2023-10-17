"""Stores memories of past episodes. 

If no compression is used, equivalent to a replay buffer.
"""

import torch
from typing import List


class EpisodicBuffer:
    """Remembers episodic rollouts."""

    def __init__(
        self,
        episode_length: int,
        observation_dim: int,
        action_dim: int,
    ):
        """Initialize an `EpisodicBuffer`.

        Args:
            episode_length: Number of timesteps per episode.
            observation_dim: Dimension of observations.
            action_dim: Dimension of actions.
        """
        self.episode_length = episode_length
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.observations: List[torch.Tensor] = []
        """List of observations, containing floating-point tensors of shape [T, O]."""
        self.actions: List[torch.Tensor] = []
        """List of actions, containing tensors of shape [T, A]."""
        self.rewards: List[torch.Tensor] = []
        """List of rewards, containing floating-point tensors of shape [T]."""
        self.dones: List[torch.Tensor] = []
        """List of termination flags, containing boolean tensors of shape [T]."""

    @property
    def t(self) -> int:
        return self.episode_length

    @property
    def o(self) -> int:
        return self.observation_dim

    @property
    def a(self) -> int:
        return self.action_dim

    def add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Add data from an episode to the `EpisodicBuffer`.

        Args:
            observations: _description_
            actions: _description_
            rewards: _description_
            dones: _description_
        """
        assert observations.shape == (
            self.t,
            self.o,
        ), f"Episode observations had wrong shape (expected ({self.t, self.o}), got ({observations.shape}))."
        assert (
            observations.dtype.is_floating_point
        ), "Episode observations must be floating-point."
        self.observations.append(observations)

        assert actions.shape == (
            self.t,
            self.a,
        ), f"Episode actions had wrong shape (expected ({self.t, self.a}), got ({actions.shape}))."
        self.actions.append(actions)

        assert rewards.shape == (
            self.t,
        ), f"Episode rewards had wrong shape (expected ({self.t}), got ({rewards.shape}))."
        assert (
            rewards.dtype.is_floating_point
        ), "Episode rewards must be floating-point."
        self.rewards.append(rewards)

        assert dones.shape == (
            self.t,
        ), f"Episode dones had wrong shape (expected ({self.t}), got ({dones.shape}))."
        assert dones.dtype == torch.bool, "Episode dones must be boolean."
        self.dones.append(dones)

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
