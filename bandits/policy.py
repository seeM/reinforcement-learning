"""
TODO: Write tests.
"""

import numpy as np


class Policy():
    pass

class GreedyPolicy(Policy):
    """
    The greedy policy always chooses the greedy action, i.e., the action with
    highest percieved (or estimated) value. Ties are settled uniformly randomly.   
    """
    # We can't just use np.argmax(), since that only returns
    # the first occurrence of the maximum
    def decide(self, values):
        """
        Decide on an action, based on their percieved values.

        Parameters
        ----------
        values : np.ndarray of length n_actions
            Estimated (or percieved) values of each action
        
        Returns
        -------
        action : int
            The chosen (greedy) action to take
        """
        max_value = np.max(values)
        optimal_actions = np.argwhere(values == max_value).reshape(-1)

        return np.random.choice(optimal_actions)

class EpsilonGreedyPolicy(Policy):
    """
    The epsilon-greedy policy 'explores' with probability `eps`, i.e., it
    uniformly randomly decides on an action, otherwise with probability
    1 - `eps` it 'exploits' the greedy action, i.e., the action with highest
    probability. Ties are settled uniformly randomly.
    """
    def __init__(self, eps):
        self.eps = eps

    def decide(self, values):
        """
        Decide on an action, based on their percieved values.

        Parameters
        ----------
        values : np.ndarray of length n_actions
            Estimated (or percieved) values of each action
        
        Returns
        -------
        action : int
            The chosen (greedy) action to take
        """
        explore = np.random.choice([True, False], p=[self.eps, 1 - self.eps])
        
        if explore:
            return np.random.randint(0, len(values))
        else:
            max_value = np.max(values)
            optimal_actions = np.argwhere(values == max_value).reshape(-1)

            return np.random.choice(optimal_actions)
