from abc import ABC, abstractmethod

import torch

from rl_box.policies.params import CategoricalDistributionParams


class ActionDistribution(ABC):
    @property
    def a(self):
        return self.action_dim

    @property
    def d(self):
        return self.params_dim()

    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of actions sampled from this distribution."""
        raise NotImplementedError

    @abstractmethod
    def params_dim(self) -> int:
        """Dimension of the parameterization of this distribution."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, params: torch.Tensor) -> torch.Tensor:
        """Sample actions from a distribution with the given parameters.

        Args:
            params: Tensor with shape [*, D].

        Returns:
            actions: Tensor with shape [*, A].
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(
        self, params: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Get the log-probability of samples under a distribution with the given parameters.

        Args:
            params: Tensor with shape [*, D].
            actions: Tensor with shape [*, A].

        Returns:
            log_probs: Tensor with shape [*].
        """
        raise NotImplementedError

    @abstractmethod
    def entropy(self, params: torch.Tensor) -> torch.Tensor:
        """Get the entropy of the distribution with the given parameters.

        Args:
            params: Tensor with shape [*, D].

        Returns:
            actions: Tensor with shape [*].
        """
        raise NotImplementedError


class CategoricalDistribution(ActionDistribution):
    def __init__(self, params: CategoricalDistributionParams):
        self.num_classes = params.num_classes

    def action_dim(self) -> int:
        return 1

    def params_dim(self) -> int:
        return self.num_classes

    def sample(self, params: torch.Tensor) -> torch.Tensor:
        return (
            torch.distributions.Categorical(logits=params).sample().view(-1, 1)
        )

    def log_prob(
        self, params: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        logits = params - params.logsumexp(dim=-1, keepdim=True)
        action = action.squeeze(-1)
        return logits.gather(-1, action)

    def entropy(self, params: torch.Tensor) -> torch.Tensor:
        logits = params - params.logsumexp(dim=-1, keepdim=True)
        min_real = torch.finfo(logits.dtype).min
        logits = torch.clamp(logits, min=min_real)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return -(logits * probs).sum(-1)
