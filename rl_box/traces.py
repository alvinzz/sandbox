"""Compute "eligability traces" for temporal credit assignment in RL.
"""


from typing import Tuple

import torch


def lambda_returns_and_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the episodic \lambda-returns and \lambda-advantage.

    Uses many ideas from TD-\lambda [1].

    Args:
        rewards: Tensor of empirical episodic rewards, with shape [*, T].
        dones: Tensor of termination flags, with shape [*, T].
        values: Tensor of estimated expected returns, with shape [*, T + 1].
        gamma: Temporal discount factor, in the range [0.0, 1.0].
        lambda_: Trace-decay parameter, in the range [0.0, 1.0].

    Returns:
        lambda_returns: Tensor of episodic \lambda-returns, with shape [*, T].
        lambda_advantages: Tensor of episodic \lambda-advantages, with shape [*, T].

    Details:
    1) Overview of \lambda-return:

        The 1-step return R_t^1 is:
            r_t + \gamma * V(s_{t+1})

        The 2-step return R_t^2 is:
            r_t + \gamma * r_{t+1} + \gamma^2 * V(s_{t+2})

        As n increases, the variance of R_t^n increases.
        But, the bias (for imperfect value estimators V) decreases.

        The "\lambda-return" R_t^\lambda is a weighted average of these:
            R_t^\lambda = (R_t^1 + \lambda * R_t^2 + \lambda^2 * R_t^3 + ...) / (1 + \lambda + \lambda^2 + ...)

        As \lambda increases, R_t^n with higher n's are weighted more.
        Thus the variance increases and the bias decreases.

    2) Overview of \lambda-advantage:

        Gradient policy-improvement methods directly optimize the parameters
        of the policy against the expected return.

        The policy gradient theorem says:
            ∇_theta E[ \sum_t [ r_t ] ] = E[ \sum_t [ \Psi_t * ∇_theta (\log \pi_theta(a_t | s_t)) ] ],
        where \Psi_t = Q^\pi(s_t, a_t) - b(s_t).

        When b(s_t) is taken to be V(s_t), \Psi_t is the advantage function
            A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)
              = E[ R_t^\lambda ] - V^\pi(s_t),
        when the \lambda-return is computed using V = V^\pi.

        GAE [2] proposes using the episodic \lambda-return even for approximate V =/= V^pi.
        They suggest that it provides a good bias-variance trade-off and helps with credit assignment.

        Explicitly, they use:
            \Psi_t = A(s_t, a_t) = R_t^\lambda - V(s_t).

        Empirically, it works well and is used in practically all policy-gradient implementations.

    References:
        [1] : http://incompleteideas.net/book/ebook/node73.html
        [2] : Schulman et. al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation, ICLR 2016. https://arxiv.org/abs/1506.02438
    """
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("`gamma` must be in the range [0.0, 1.0].")

    if lambda_ < 0.0 or lambda_ > 1.0:
        raise ValueError("`lambda_` must be in the range [0.0, 1.0].")

    assert rewards.dtype.is_floating_point, "Rewards must be floating-point."
    assert dones.dtype == torch.bool, "Dones must be boolean."
    assert values.dtype.is_floating_point, "Values must be floating-point."

    batch_shape = rewards.shape[:-1]
    episode_len = rewards.shape[-1]
    assert (
        rewards.shape == dones.shape
    ), f"`rewards` and `dones` must have the same shape, got {rewards.shape} and {dones.shape}."
    assert (
        values.shape[:-1] == batch_shape
    ), f"`rewards` and `values` must have the same shape (excluding time dimension), got {rewards.shape[:-1]} and {values.shape[:-1]}."
    assert (
        values.shape[-1] == episode_len + 1
    ), f"`values` must have one more (time) observation than `rewards`, got {rewards.shape[:-1]} and {values.shape[:-1]}."

    not_dones = ~dones
    not_dones = not_dones.int()

    lambda_returns = torch.empty(
        rewards.shape, device=rewards.device, dtype=rewards.dtype
    )

    last_t = episode_len - 1
    lambda_returns[..., last_t] = (
        rewards[..., last_t]
        + gamma * not_dones[..., last_t] * values[..., last_t + 1]
    )

    # Note:
    # R_t^n = r_t + ... + \gamma^{n-1} * r_{t+n-1} + \gamma^n * V(s_{t+n})
    #       = r_t + \gamma * R_{t+1}^{n-1}
    # R_t^\lambda = (R_t^1 + \lambda * R_t^2 + \lambda^2 * R_t^3 + ...) / (1 + \lambda + \lambda^2 + ...)
    #             = [(r_t + \gamma * V(s_{t+1}) + \lambda * (r_t + \gamma * R_{t+1}^1)) + \lambda^2 * (r_t + \gamma * R_{t+1}^2) ] / (1 + \lambda + \lambda^2 + ...)
    #             = r_t + \gamma * V(s_{t+1}) / (1 + \lambda + \lambda^2 + ...) + \lambda * \gamma * R_{t+1}^\lambda
    #             = r_t + (1.0 - \lambda) * \gamma * V(s_{t+1}) + \lambda * \gamma * R_{t+1}^\lambda
    for t in reversed(range(episode_len - 1)):
        lambda_returns[..., t] = (
            rewards[..., t]
            + (1.0 - lambda_) * gamma * not_dones[..., t] * values[..., t + 1]
            + lambda_ * gamma * lambda_returns[..., t + 1]
        )

    lambda_advantages = lambda_returns - values[..., :-1]

    return lambda_returns, lambda_advantages
