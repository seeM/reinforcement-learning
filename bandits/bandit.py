import numpy as np


class Bandit():
    """
    A k-armed bandit problem with stationary, unit-variance Gaussian-
    distributed rewards and randomly generated Gaussian-distributed values.

    Parameters
    ----------
    n_arms : int
        The number of arms (or actions)

    Attributes
    ----------
    n_arms : int
        The number of arms (or actions)
    values : array of shape (n_arms, 1)
        The value of each action
    """

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.values = np.random.normal(loc=0, scale=1, size=n_arms)
        self.optimal_action = np.argmax(self.values)  # assuming a single max

    def reward(self, action):
        """
        Sample the reward of a given action from a Gaussian distribution with
        mean given by `values` and variance of one.

        Parameters
        ----------
        action : int
            The index of the action taken

        Return
        ------
        reward : float
            The value of the reward recieved
        """
        return np.random.normal(loc=self.values[action], scale=1)
