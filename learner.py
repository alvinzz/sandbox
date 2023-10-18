from abc import ABC, abstractmethod
from typing import Generic, Iterable, Callable, List

from data import Data

import torch


class Learner(ABC, Generic[Data]):
    """Abstract class for a structure that learns from `Data`.

    Args:
        Generic: Type of `Data`.
    """

    @abstractmethod
    def update(self, new: Iterable[Data]):
        """Update the `Learner` using new `Data`.

        Args:
            new: `Data` to learn from.
        """
        raise NotImplementedError


class GradientLearner(Learner[Data]):
    """A `Learner` that performs gradient descent on an objective function."""

    def __init__(
        self,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[[Data], torch.Tensor],
    ):
        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function

    def update(self, new: Iterable[Data]):
        """Perform gradient descent on new data.

        Args:
            new: `Data` to learn from.
        """
        self.optimizer.zero_grad()

        losses: List[torch.Tensor] = []
        for datum in new:
            losses.append(self.loss_function(datum))
        loss: torch.Tensor = torch.stack(losses, dim=0).mean(dim=0)
        loss.backward()

        self.optimizer.step()
