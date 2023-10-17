from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from environment import Environment

Observation = TypeVar("Observation")
Action = TypeVar("Action")


class Agent(ABC, Generic[Observation, Action]):
    """An autonomously acting unit in an `Environment`.

    Args:
        Observation: Type of the `Agent`'s state.
        Action: Type of the `Agent`'s interaction with its `Environment`.

    Attributes:
        environment: `Environment` that the `Agent` operates in.
    """

    def act(self, environment: Environment) -> Action:
        """`learn`, `sense`, and `plan`.

        Returns:
            action: Action that the `Agent` takes at this time(step).
        """
        observation = self.observe(environment)
        self.learn(observation)
        return self.plan()

    @abstractmethod
    def observe(self, environment: Environment) -> Observation:
        """Get measurements from the `Agent`'s sensors.

        Returns:
            observation: State of the `Environment` accessible to the `Agent`.
        """
        return NotImplementedError

    @abstractmethod
    def learn(self, observation: Observation):
        """Update the `Agent`'s internal state.

        Args:
            observation: The `Agent`'s latest `Observation`.
        """
        return NotImplementedError

    @abstractmethod
    def plan(self) -> Action:
        """Plan to take an `Action` in an `Environment`.

        Returns:
            action: The `Action` that the `Agent` will take at this time.
        """
        return NotImplementedError
