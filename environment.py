from abc import ABC, abstractmethod


class Environment(ABC):
    """A slice of the world."""

    @abstractmethod
    def step(self):
        """Advance the `Environment` by one timestep.

        Should account for the `Action` of every `Agent` in the `Environment`.
        """
