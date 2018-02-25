import numpy as np
import matplotlib.pyplot as plt

from bandits.agent import Agent
from bandits.bandit import Bandit
from bandits.policy import GreedyPolicy, EpsilonGreedyPolicy
from bandits.environment import Environment

def run_simple(n_actions, n_steps):
    bandit = Bandit(n_actions)
    agent = Agent(n_actions, GreedyPolicy())

    rewards = np.empty(n_steps)

    env = Environment(bandit, agent)
    env.simulate(n_steps)
    rewards = agent.rewards

    plt.plot(rewards)
    plt.show()

def run_many_bandits(n_actions, n_steps, n_runs):
    bandits = [Bandit(n_actions) for _ in range(n_runs)]
    agent = Agent(n_actions, GreedyPolicy())

    rewards = np.empty((n_steps, n_runs))

    for i, bandit in enumerate(bandits):
        agent.forget()
        envir = Environment(bandit, agent)
        envir.simulate(n_steps)
        rewards[:, i] = agent.rewards

    plt.plot(rewards.mean(axis=1))
    plt.show()

def run_many_bandits_many_agents(n_actions, n_steps, n_runs):
    bandits = [Bandit(n_actions) for _ in range(n_runs)]
    agents = [
        Agent(n_actions, GreedyPolicy()),
        Agent(n_actions, EpsilonGreedyPolicy(0.1))
    ]

    rewards = np.empty((n_steps, n_runs, len(agents)))

    for j, agent in enumerate(agents):
        for i, bandit in enumerate(bandits):
            agent.forget()
            envir = Environment(bandit, agent)
            envir.simulate(n_steps)
            rewards[:, i, j] = agent.rewards

    plt.plot(rewards.mean(axis=1))
    plt.show()

if __name__ == "__main__":
    n_actions = 10
    n_steps = 1000
    n_runs = 2000

    bandits = [Bandit(n_actions) for _ in range(n_runs)]
    agents = [
        Agent(n_actions, GreedyPolicy()),
        Agent(n_actions, EpsilonGreedyPolicy(0.1)),
        Agent(n_actions, EpsilonGreedyPolicy(0.01))
    ]

    rewards = np.empty((n_steps, n_runs, len(agents)))

    for j, agent in enumerate(agents):
        for i, bandit in enumerate(bandits):
            agent.forget()
            envir = Environment(bandit, agent)
            envir.simulate(n_steps)
            rewards[:, i, j] = agent.rewards

    plt.plot(rewards.mean(axis=1))
    plt.show()
