from gymnasium.envs.toy_text import CliffWalkingEnv
import numpy as np


class ModifiedCliffWalkingEnv(CliffWalkingEnv):
	def _calculate_transition_prob(self, current, delta):
		"""
		Determine the outcome for an action deterministically. If the current 
		state is in the cliff, the agent cannot transition out with any action. 
		Getting to the goal gets 1 reward. Falling off the cliff gets -1 reward.
		
		Args:
			current: Current position on the grid as (row, col)
			delta: Change in position for transition

		Returns:
			List of tuple of ``(transition_prob, new_state, reward, terminated)``
		"""

		current_state = np.ravel_multi_index(current, self.shape)


		if self._cliff[current]:
			return [(1.0, current_state, -1, True)]

		new_position = np.array(current) + np.array(delta)

		new_position = self._limit_coordinates(new_position).astype(int)
		new_state = np.ravel_multi_index(tuple(new_position), self.shape)

		if self._cliff[tuple(new_position)]:
			return [(1.0, new_state, -1, True)]

		terminal_state = (self.shape[0] - 1, self.shape[1] - 1)

		if tuple(new_position) == terminal_state:
			return [(1.0, new_state, 1, True)]

		else:
			return [(1.0, new_state, 0, False)]
		

class StochasticCliffWalkingEnv(CliffWalkingEnv):
	def _calculate_transition_prob(self, current, delta):
		"""
		Determine the outcome for an action stochastically. With probability 
		0.8, the agent transitions to the position that its action intends to.
		With uniform probability 0.05, it transitions to a position that 
		corresponds to a uniformly random action. The reward is the same as the
		previous setting.

		Args:
			current: Current position on the grid as (row, col)
			delta: Change in position for transition

		Returns:
			List of tuple of ``(transition_prob, new_state, reward, terminated)``
		"""


		
		big_tuple=[]
		all_delta=[[0,1],[0,-1],[1,0],[-1,0]]
		current_state = np.ravel_multi_index(current, self.shape)
		pos_delta=[ i for i in all_delta if i!=delta]


		new_position = np.array(current) + np.array(delta)

		new_position = self._limit_coordinates(new_position).astype(int)
		new_state = np.ravel_multi_index(tuple(new_position), self.shape)
		terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
		if self._cliff[current]:
			big_tuple.append((0.85, current_state, -1, True))

		

		elif self._cliff[tuple(new_position)]:
			 big_tuple.append((1, new_state, -1, True))

		

		elif tuple(new_position) == terminal_state:
			big_tuple.append((0.85, new_state, 1, True))

		else:
			big_tuple.append((0.85, new_state, 0, False))		
		
		for new_delta in pos_delta:

			new_position = np.array(current) + np.array(new_delta)

			new_position = self._limit_coordinates(new_position).astype(int)
			new_state = np.ravel_multi_index(tuple(new_position), self.shape)
			terminal_state = (self.shape[0] - 1, self.shape[1] - 1)


			if self._cliff[current]:
				big_tuple.append((0.05, current_state, -1, True))
			
			elif self._cliff[tuple(new_position)]:
				 big_tuple.append((0.0, new_state, -1, True))


			elif tuple(new_position) == terminal_state:
				big_tuple.append((0.05, new_state, 1, True))

			else:
				big_tuple.append((0.05, new_state, 0, False))


		return big_tuple








	