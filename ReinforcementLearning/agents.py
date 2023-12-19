import gymnasium as gym

import policy_iteration as pi
import value_iteration as vi
import numpy as np


class Agent:
	def __init__(self, env: gym.Env):
		self.env = env

	def learn(self, gamma: float) -> None:
		pass

	def sample(self, state: int) -> int:
		raise NotImplementedError

class RandomAgent(Agent):
	def sample(self, state: int) -> int:
		"""Take a random action."""
		return self.env.action_space.sample()


class PolicyAgent(Agent):
	def learn(self, gamma: float) -> None:
		"""Learn policy."""
		self.policy, self.value_func, n_improves, n_evals = pi.policy_iteration(
			self.env, gamma)
		print(f"Total number of policy evaluations: {n_evals}")
		print(f"Total number of policy improvements: {n_improves}")

	def sample(self, state: int) -> int:
		"""Take an action according to the policy."""
		assert self.policy is not None, \
		"No policy, learn_policy() must be invoked before sampling action."

		#takes a single state and return action according to self.policy
		return self.policy[state]

			


class ValueAgent(Agent):
	def learn(self, gamma: float) -> None:
		self.gamma = gamma
		self.value_func, n_iters = vi.value_iteration(self.env, gamma)
		print(f"Total number of value iterations: {n_iters}")

	def sample(self, state: int) -> int:
		"""Take an action that maximizes the expected value under a specific 
		state."""
		assert self.value_func is not None, \
		"No value_func, learn_value() must be invoked before sampling action."

		rhs_big=[]
		for action in self.env.P[state].keys():
			reward=self.env.P[state][action][0][2]
			ts_prob=self.env.P[state][action][0][0]
			next_state=self.env.P[state][action][0][1]
			rhs = ts_prob*(reward+ (self.gamma*self.value_func[next_state]))
			rhs_big.append(rhs)



		

		return np.argmax(rhs_big)