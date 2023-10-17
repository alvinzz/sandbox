from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Iterable

Data = TypeVar("Data")


class Dataset(ABC, Generic[Data]):
    """Abstract class for a structure that contains `Data`.

    Args:
        Generic: Type of `Data`.
    """

    @abstractmethod
    def add(self, new: Iterable[Data]):
        """Add new `Data` to the `Dataset`.

        Args:
            new: `Data` to add.
        """
        raise NotImplementedError

    @abstractmethod
    def empty(self):
        """Empty the `Dataset`."""
        raise NotImplementedError

    @abstractmethod
    def sample_n(self, n: int) -> Iterable[Data]:
        """Sample `n` `Data` points.

        This method may modify the dataset.

        Args:
            n: Number of `Data` points to sample.

        Returns:
            data: An `Iterable` containing `n` `Data` points.
        """
        raise NotImplementedError
