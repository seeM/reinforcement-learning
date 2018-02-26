import numpy as np
import matplotlib.pyplot as plt

from bandits.agent import Agent
from bandits.bandit import Bandit
from bandits.policy import GreedyPolicy, EpsilonGreedyPolicy
from bandits.environment import Environment


def k_armed_testbed(n_actions, n_steps, n_runs):
    bandits = [Bandit(n_actions) for _ in range(n_runs)]
    agents = [
        Agent(n_actions, GreedyPolicy()),
        Agent(n_actions, EpsilonGreedyPolicy(0.1)),
        Agent(n_actions, EpsilonGreedyPolicy(0.01))
    ]
    n_agents = len(agents)

    actions = np.empty((n_steps, n_runs, n_agents))
    rewards = np.empty((n_steps, n_runs, n_agents))

    for j, agent in enumerate(agents):
        for i, bandit in enumerate(bandits):
            agent.forget()
            envir = Environment(bandit, agent)
            envir.simulate(n_steps)

            actions[:, i, j] = agent.actions
            rewards[:, i, j] = agent.rewards

    optimal_actions = np.array([bandit.optimal_action for bandit in bandits])

    return actions, rewards, optimal_actions


def plot_average_reward(rewards, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    ax.plot(rewards.mean(axis=1), **kwargs)

    return ax


def plot_percent_optimal_action(actions, optimal_actions, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    n_agents = actions.shape[2]

    # Reshape optimal_actions so that element-wise boolean comparison works.
    optimal_actions = optimal_actions.reshape((1, -1))
    optimal_actions = np.stack([optimal_actions] * n_agents, axis=2)
    correct_actions = actions == optimal_actions

    ax.plot(correct_actions.mean(axis=1), **kwargs)
    ax.set_ylim(0, 1)

    return ax


if __name__ == "__main__":
    actions, rewards, optimal_actions = k_armed_testbed(10, 1000, 2000)
