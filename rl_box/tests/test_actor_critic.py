from typing import Tuple

import pytest
import torch

from rl_box.policies.actor_critic import ActorCritic
from rl_box.policies.config import *


@pytest.mark.parametrize("normalize_advantages", [True, False])
@pytest.mark.parametrize("arm_rewards", [(0.33, 0.67)])
def test_actor_critic(
    normalize_advantages: bool, arm_rewards: Tuple[float, float]
):
    """Test ActorCritic on a 2-arm Bandit with constant rewards."""
    torch.manual_seed(0)

    if arm_rewards[0] > arm_rewards[1]:
        arm_rewards = tuple(reversed(arm_rewards))

    B: int = 2
    T: int = 1
    O: int = 1

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.p = torch.nn.parameter.Parameter(
                torch.zeros((3,), dtype=torch.float32),
            )

        def forward(self, x: torch.Tensor):
            result = self.p
            for _ in range(x.dim() - 1):
                result = result.unsqueeze(0)
            return result.broadcast_to(*x.shape[:-1], 3)

    net = Net()
    optim = torch.optim.Adam(net.parameters(), 0.01)
    config = ActorCriticConfig(
        actor=ActorConfig(
            distribution=CategoricalDistributionConfig(num_classes=2),
            algo=PpoConfig(clip_coef=0.1),
            normalize_advantages=normalize_advantages,
        ),
        critic=CriticConfig(huber_loss_coef=1.0),
        gamma=1.0,
        lambda_=1.0,
        actor_coef=1.0,
        critic_coef=1.0,
    )

    policy = ActorCritic(1, 1, net, optim, config)

    observations = torch.zeros((B, T + 1, O), dtype=torch.float32)
    for _ in range(1000):
        action_distribution_params, value_estimates = policy.forward(
            observations
        )
        action_distribution_params = action_distribution_params[..., :-1, :]
        actions = policy.action_distribution.sample(action_distribution_params)
        rewards = (
            actions * arm_rewards[1] + (1 - actions) * arm_rewards[0]
        ).squeeze(-1)
        dones = torch.ones((B, T), dtype=torch.bool)
        data = (
            observations,
            value_estimates,
            action_distribution_params,
            actions,
            rewards,
            dones,
        )
        policy.update(data)

    assert (
        torch.softmax(net.p[:2], -1)[1]
        > torch.softmax(torch.tensor(arm_rewards), -1)[1]
    ), "bad policy"
    assert (
        abs(value_estimates[0, 0].item() - arm_rewards[1]) < 0.01
    ), "bad value estimate"
