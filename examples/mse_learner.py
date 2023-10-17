from typing import Iterable

import torch

from data import Dataset
from learner import Learner


Data = torch.Tensor
"""A Tensor with shape [2], containing an (x, y) pair.
"""


class NoisyLinearDataset(Dataset[Data]):
    """Sample dataset.

    Linear with Gaussian noise added:

    `y = mx + b + n`,
    where:
        `n ~ N(0, sigma)`
    """

    def __init__(self, m: float, b: float, sigma: float, size: int):
        """Create a `NoisyLinearDataset`.

        Args:
            m: m
            b: b
            sigma: sigma
            size: Number of samples in the dataset.
        """
        if sigma < 0.0:
            raise ValueError("`sigma` must be nonnegative.")

        if size < 2:
            raise ValueError("`size` must be at least 2.")

        self.m: float = m
        self.b: float = b
        self.sigma: float = sigma
        self.size: int = size

        x = torch.rand(size)  # [size]
        n = torch.normal(0.0, sigma, (size,))  # [size]
        y = m * x + b + n  # [size]

        self.x: torch.Tensor = x
        self.y: torch.Tensor = y

    def add(self, new: Iterable[Data]):
        raise NotImplementedError

    def empty(self):
        raise NotImplementedError

    def sample_n(self, n: int) -> Iterable[Data]:
        raise NotImplementedError

    def all(self) -> torch.Tensor:
        """Get all `Data` from `Self`.

        Returns:
            data: A `torch.Tensor` with shape [`self.size`, 2].
        """
        return torch.stack((self.x, self.y), dim=1)


class MSELearner(Learner[Data]):
    """A simple `Learner` for the `NoisyLinearDataset`.

    Attempts to model the `NoisyLinearDataset` as:
      `y = mx + b`,
    where `m` and `b` are the learnable parameters.

    The update performs GD on the (mean) squared error*:
      `L(m, b | x, y) = 0.5 * (y - mx - b)^2`

      `m_update = x * (y - mx - b)`

      `b_update = y - mx - b`

    *This also maximizes the log-likelihood of the data under the assumption of
    i.i.d. zero-mean Gaussian noise.
    """

    def __init__(self, learning_rate: float):
        """Create a new `MSELearner`.

        Args:
            learning_rate: Gradient descent step-size.
        """
        self.lr = learning_rate
        self.m = 0.0
        self.b = 0.0

    def update(self, new: Iterable[Data]):
        """Perform one gradient step on the `MSELearner`.

        Args:
            new: `Data` in the form of a `torch.Tensor` ([B, 2], f32).
        """
        if (
            not isinstance(new, torch.Tensor)
            or new.shape[1] != 2
            or new.dtype != torch.float32
        ):
            raise ValueError(
                "`new` `Data` must be a torch.Tensor with shape [B, 2]."
            )

        x = new[..., 0]  # [B]
        y = new[..., 1]  # [B]

        prediction = self.m * x + self.b  # [B]
        error = y - prediction  # [B]

        b_update = torch.mean(error).cpu().numpy()  # average over "batch"
        self.b += self.lr * b_update

        m_update = torch.mean(x * error).cpu().numpy()  # average over "batch"
        self.m += self.lr * m_update

    def mean_squared_error(self, data: torch.Tensor) -> float:
        """Compute the mean squared error for a batch of `Data`.

        Args:
            data: `Data` in the form of a `torch.Tensor` ([B, 2], f32).

        Returns:
            mse: The mean squared error.
        """
        x = data[..., 0]
        y = data[..., 1]

        prediction = self.m * x + self.b  # [B]
        error = y - prediction  # [B]

        return torch.mean(error**2).cpu().numpy()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # torch.manual_seed(0)
    plt.ion()

    dataset = NoisyLinearDataset(0.5, 1.0, 0.2, 50)
    learner = MSELearner(0.1)

    data = dataset.all()

    xs = np.linspace(0.0, 1.0, 100)

    for epoch in range(100):
        learner.update(data)

        error = learner.mean_squared_error(data)

        print(
            f"Epoch {epoch}: mse {error:.2f}, m {learner.m:.2f}, b {learner.b:.2f}."
        )

        plt.plot(xs, learner.m * xs + learner.b)
        plt.scatter(data[..., 0], data[..., 1])
        plt.savefig(f"/tmp/{epoch}.png")
        plt.close()
