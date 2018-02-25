import numpy as np

from policy import *
from agent import *

greedy = GreedyPolicy()
eps_greedy = EpsilonGreedyPolicy(0.1)

"""
Greedy policy tests.
"""

"""
A simple test with a clear maximum.
"""
test = [5, 3, 6, 7, 2]

assert greedy.decide(test) == 3

"""
A test with two maxima.
"""
test = [5, 3, 6, 7, 7]

results = np.array([greedy.decide(test) for _ in range(1000)])

assert (results == 3).sum() / 1000 > 0.4
assert (results == 4).sum() / 1000 > 0.4

"""
EpsilonGreedy policy tests.
"""
test = [5, 3, 6, 7, 2]

