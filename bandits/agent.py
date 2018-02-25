"""
TODO: Add asserts to last_* functions to ensure length is >= 1.
"""

import numpy as np


class Agent():
	"""
	An agent, based on its values, acts in an environment, recieves
	rewards for its actions, and updates its values based on a policy.

	Parameters
	----------
	policy : Policy object
		The policy by which the agent decides on actions
	n_actions : integer
		The number of actions to decide between

	Attributes
	----------
	values : array of length n_actions
		The values estimated by the agent
	actions : list
		The actions taken by the agent
	rewards : list
		The rewards recieved for each action
	"""
	def __init__(self, n_actions, policy):
		self.n_actions = n_actions
		self.policy = policy
		self.values = np.zeros(n_actions)
		self.actions_taken = np.zeros(n_actions, dtype=np.int64)
		self.actions = []
		self.rewards = []

	@property
	def last_action(self):
		"""
		The last action taken by the agent.

		Return
		------
		action : integer
			The index of the action
		"""
		# if len(self.actions) == 0:
		# 	return 0
		# else:
		# 	return self.actions[-1]
		return self.actions[-1]

	@property
	def last_reward(self):
		"""
		The last reward recieved by the agent.

		Return
		------
		reward : float
			The reward value
		"""
		return self.rewards[-1]

	def act(self):
		"""
		An agent acts based on its values and its policy.

		Returns
		-------
		action : integer in the interval [1, n_actions]
			The index of the decided action
		"""
		self.actions.append(self.policy.decide(self.values))
		self.actions_taken[self.last_action] += 1
		return self.last_action

	def update_values(self, reward):
		"""
		Update the agent's values

		TODO: Implement different update rules.
		TODO: Allow customization of the weighting parameter.

		Parameters
		----------
		reward : float
			The reward value

		Returns
		-------
		values : array of length n_actions
			The values estimated by the agent
		"""
		self.rewards.append(reward)
		self.values[self.last_action] += (
			1 / self.actions_taken[self.last_action]
			* (self.last_reward - self.values[self.last_action])
		)

		return self.values

	def forget(self):
		"""
		Forget past actions and rewards, and reset actions taken and estimated
		values to zero.

		Returns
		-------
		agent : Agent object
			A copy of the agent object
		"""
		self.actions = []
		self.rewards = []
		self.values = np.zeros_like(self.values)
		self.actions_taken = np.zeros_like(self.actions_taken)

		return self
