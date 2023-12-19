from typing import Tuple

import gymnasium as gym
import numpy as np


def value_iteration(
		env: gym.Env, 
		gamma: float, 
		max_iterations: int = int(1e3), 
		tol: float = 1e-3) -> Tuple[np.ndarray, int]:
	"""Runs value iteration until value function converges.

	Args:
		env: Gym environment
		gamma: Discount factor 
		max_iterations: Max number of iterations before stopping
		tol: Tolerance for convergence
	
	Returns:
		Learned value function (1D np.ndarray of shape (48) containing the estimated value at each state)
		Number of value iterations (int containing the number of iterations until value iteration converged or returned)
	"""
	value_func = np.zeros(env.nS) 
	for t in range(0,max_iterations):
		delta=0
		# rhs_big=[]

		for s in range(env.nS):
		# old_action=policy[s]
			rhs_big=[]
			new_v=value_func[s]
			# rhs=0

			
			for action in env.P[s].keys():
				tup=env.P[s][action]
				rhs=0
				# print("tup: ",tup,action)
				for i in range(len(tup)):
					ts_prob=tup[i][0]
					new_state=tup[i][1]
					reward=tup[i][2]
					rhs += ts_prob*(reward+ (gamma*value_func[new_state]))

					# print("RHS: ",rhs)
				rhs_big.append(rhs)
			value_func[s]=max(np.array(rhs_big))
			delta=max(delta,np.abs(new_v-value_func[s]))
		if delta<tol:
			break
	return (value_func,t)











