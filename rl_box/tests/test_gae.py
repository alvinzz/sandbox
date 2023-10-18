from typing import Tuple

import torch

from rl_box.traces import lambda_returns_and_advantages


def torch_gae(
    gamma: float,
    lmbda: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    terminated: None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """torchrl's implementation, with the time-dim shifted by one."""
    if terminated is None:
        terminated = done
    if not (
        next_state_value.shape
        == state_value.shape
        == reward.shape
        == done.shape
        == terminated.shape
    ):
        raise RuntimeError("shape_err")
    dtype = next_state_value.dtype
    device = state_value.device
    not_done = (~done).int()
    not_terminated = (~terminated).int()
    *batch_size, time_steps = not_done.shape
    advantage = torch.empty(
        *batch_size, time_steps, device=device, dtype=dtype
    )
    prev_advantage = 0
    g_not_terminated = gamma * not_terminated
    delta = reward + (g_not_terminated * next_state_value) - state_value
    discount = lmbda * gamma * not_done
    for t in reversed(range(time_steps)):
        prev_advantage = advantage[..., t] = delta[..., t] + (
            prev_advantage * discount[..., t]
        )

    value_target = advantage + state_value

    return advantage, value_target


def test_gae():
    lambda_ = 0.5
    gamma = 0.8

    values = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )
    rewards = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    )
    dones = torch.tensor(
        [
            [True, True, True],
            [False, True, True],
            [False, False, True],
            [False, False, False],
        ]
    )

    torch_adv, torch_return = torch_gae(
        gamma,
        lambda_,
        values[..., :-1],
        values[..., 1:],
        rewards,
        dones,
    )
    my_return, my_adv = lambda_returns_and_advantages(
        rewards, dones, values, gamma, lambda_
    )

    assert torch.allclose(torch_return, my_return)
    assert torch.allclose(torch_adv, my_adv)
