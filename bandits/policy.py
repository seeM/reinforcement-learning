import numpy as np


def greedy(values):
    """
    The greedy algorithm always chooses the greedy action,
    that is, the action with highest percieved (or estimated) value.
    Ties are settled (uniformly) randomly.
    
    Parameters
    ----------
    values : np.ndarray
        Estimated (or percieved) values of each action
    
    Returns
    -------
    action : int
        The chosen (greedy) action to take    
    """
    # We can't just use np.argmax(), since that only returns
    # the first occurrence of the maximum
    max_value = np.max(values)
    optimal_actions = np.argwhere(values == max_value).reshape(-1)

    return np.random.choice(optimal_actions)


def eps_greedy(values, eps):
    """
    The greedy algorithm always chooses the greedy action,
    that is, the action with highest percieved (or estimated) value.
    Ties are settled (uniformly) randomly.
    
    Parameters
    ----------
    values : np.ndarray
        Estimated (or percieved) values of each action
    
    Returns
    -------
    action : int
        The chosen (greedy) action to take    
    """
    explore = np.random.choice([True, False], p=[eps, 1 - eps])
    
    if explore:
        return np.random.randint(0, len(values))
    else:
        max_value = np.max(values)
        optimal_actions = np.argwhere(values == max_value).reshape(-1)

        return np.random.choice(optimal_actions)
