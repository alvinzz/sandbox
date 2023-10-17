from abc import ABC, abstractmethod
from typing import Generic, Iterable

from data import Data


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
