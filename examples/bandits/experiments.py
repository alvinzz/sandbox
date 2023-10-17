import numpy as np

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from examples.bandits.environments.bernoulli import BernoulliArms
    from examples.bandits.environments.gaussian import GaussianArms
    from examples.bandits.environments.mixture_of_gaussians import (
        MixtureOfGaussiansArms,
    )

    # from examples.bandits.agents.epsilon_greedy import EpsilonGreedyAgent
    # agent = EpsilonGreedyAgent(epsilon=0.1, decay=0.999)
    from examples.bandits.agents.ucb import UcbAgent
    from examples.bandits.agents.conjugate_gaussian_thompson import (
        ConjugateGaussianThompsonAgent,
    )
    from examples.bandits.agents.bootstrap_thompson import (
        BootstrapThompsonAgent,
    )

    from tqdm import tqdm

    n_trials = 1000
    trial_len = 10000
    n_arms = 10

    regret_histories = {
        "conjugate_gaussian": [],
        "bootstrap": [],
    }  # [N, T]
    for trial in tqdm(range(n_trials)):
        # agent = UcbAgent()
        agent = ConjugateGaussianThompsonAgent()

        arms = MixtureOfGaussiansArms(
            n_arms=n_arms,
            agent=agent,
            # alpha=1.0,
            # beta=1.0,
            mu_0=0.0,
            lambda_0=1.0,
            alpha_0=2.0,
            beta_0=2.0,
            dirichlet_coefs=(1.0, 1.0),
        )

        rewards = []
        for step in range(trial_len):
            arms.step()
            rewards.append(arms.last_reward)

        oracle_cum_reward = np.max(
            np.sum(np.array(arms.mixing_coefs) * np.array(arms.means), axis=1)
        ) * np.arange(trial_len)
        regret_histories["conjugate_gaussian"].append(
            oracle_cum_reward - np.cumsum(rewards)
        )

        arms.agent = BootstrapThompsonAgent()
        arms.last_reward = None
        rewards = []
        for step in range(trial_len):
            arms.step()
            rewards.append(arms.last_reward)

        oracle_cum_reward = np.max(
            np.sum(np.array(arms.mixing_coefs) * np.array(arms.means), axis=1)
        ) * np.arange(trial_len)
        regret_histories["bootstrap"].append(
            oracle_cum_reward - np.cumsum(rewards)
        )

    import pickle

    pickle.dump(regret_histories, open("regret_histories.pkl", "wb"))

    timesteps = np.arange(trial_len) + 1
    quantiles = {
        method: {
            quantile: np.quantile(history, float(quantile), axis=0)
            for quantile in (
                "0.01",
                "0.05",
                "0.25",
                "0.50",
                "0.75",
                "0.95",
                "0.99",
            )
        }
        for (method, history) in regret_histories.items()
    }
    fig, ax = plt.subplots()
    colors = ("mediumspringgreen", "fuchsia")
    for color, (method, quantiles) in zip(colors, quantiles.items()):
        ax.plot(
            timesteps,
            quantiles["0.50"],
            label=method,
            color=color,
        )
        ax.fill_between(
            timesteps,
            quantiles["0.25"],
            quantiles["0.75"],
            facecolor=color,
            alpha=0.40,
        )
        ax.fill_between(
            timesteps,
            quantiles["0.05"],
            quantiles["0.95"],
            facecolor=color,
            alpha=0.20,
        )
        ax.fill_between(
            timesteps,
            quantiles["0.01"],
            quantiles["0.99"],
            facecolor=color,
            alpha=0.10,
        )
    plt.legend()
    plt.show()
