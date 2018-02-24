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


# TODO: Make algorithms (agents?) their own classes
def run_testbed(n_actions, n_steps, algo, **kwargs):
    bandit = Bandit(n_actions)

    estimated_values = np.zeros(n_actions) # TODO: Make this and below 2D, with second dimension being timesteps
    actions_taken = np.zeros(n_actions)

    actions = np.empty(n_steps, dtype=np.int64)
    rewards = np.empty(n_steps)
    
    step = 1
    for i in range(n_steps):
        # Decide on an action
        actions[i] = algo(estimated_values, **kwargs)

        # Take the action and receive a reward
        rewards[i] = bandit(actions[i])

        # Update estimated values based on the action taken and the reward recieved
        # TODO: Make this a function
        actions_taken[actions[i]] += 1
        estimated_values[actions[i]] += 1 / (actions_taken[actions[i]]) * (rewards[i] - estimated_values[actions[i]])
    
    return actions, rewards

def run_many(n_actions, n_steps, n_runs, algo, **kwargs):
    actions = np.empty((n_steps, n_runs), dtype=np.int64)
    rewards = np.empty((n_steps, n_runs))
    
    for i in range(n_runs):
        actions[:, i], rewards[:, i] = run_testbed(n_actions, n_steps, algo, **kwargs)
    
    return actions, rewards