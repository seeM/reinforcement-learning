"""
I think it would be nice if we could have `environment` objects.
Agents take actions and recieve rewards; Environments recieve actions and give rewards.
But, there's no point storing actions and rewards in both objects.
Since we want to have many agents in a single environment,
say, one using a greedy policy and one using an epsilon-greedy policy,
the actions and rewards will be "remembered" by each agent
(rather than all by the environment).
The environment is a container for bandits, i.e., it instantiates bandits
and fixes their probability distributions.
The agents have some policy, take actions and recieve rewards, and remember all of it.

It's like randomly generating a game world, and then having players roam the world and interact with it.
In that sense, environment is like `Game`, and agents are like `Player`.
"""

import numpy as np

from bandits.agent import Agent
from bandits.bandit import Bandit


class Environment():
    """
    An environment is a container for bandits, i.e., it instantiates bandits and
    fixes their probability distributions. Agents act in environments and
    recieve rewards from environments based on those actions.

    Parameters
    ----------


    Attributes
    ----------
    n_actions : 

    bandit : 

    agent : 

    """
    def __init__(self, bandit, agent):
        assert agent.n_actions == bandit.n_arms
        self.n_actions = agent.n_actions
        self.bandit = bandit
        self.agent = agent

    def simulate(self, n_steps):
        for i in range(n_steps):
            action = self.agent.act()
            reward = self.bandit.reward(action)
            self.agent.update_values(reward)
