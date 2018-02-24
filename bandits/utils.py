import numpy as np


def running_average(x):
    """
    Calculate the running average of an array.
    
    Parameters
    ----------
    x : array
        The input array
    
    Returns
    -------
    ave : array
        The running average of x
    """
    if x.ndim == 1:
        x = x.reshape((-1,1))
    ave = np.empty(x.shape)
    
    ave[0, :] = x[0, :]
    for i in range(1, len(ave)):
        ave[i, :] = ave[i-1, :] + 1 / (i + 1) * (x[i, :] - ave[i-1,:])
    
    return ave

assert all(running_average(np.array([0, 5, 4, 9, 3])) == np.reshape([0, 2.5, 3, 4.5, 4.2], (-1, 1)))
