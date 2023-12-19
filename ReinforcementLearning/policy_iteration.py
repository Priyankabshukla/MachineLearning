from typing import Tuple

import gymnasium as gym
import numpy as np


def evaluate_policy(env: gym.Env, 
	value_func: np.ndarray, 
	gamma: float, 
	policy: np.ndarray, 
	max_iterations: int = int(1e3), 
	tol: float = 1e-3) -> Tuple[np.ndarray, int]:
	"""Performs policy evaluation.
	
	Args:
		env: Gym enviornment
		value_func: Value function indexed by state (1D np.ndarray of shape (48) containing the estimated value at each state)
		gamma: Discount factor
		policy: Policy function indexed by state (1D np.ndarray of shape (48) containing the policy at each state)
		max_iterations: Max number of iterations before stopping
		tol: tolerance for convergence

	Returns:
		Updated value function (1D np.ndarray of shape (48) containing the estimated value at each state)
		Number of policy iterations 
	"""



	for t in range(0,max_iterations):
		delta=0
		for s in range(env.nS): 
		    new_V=0  #single value

		    tup=env.P[s][policy[s]]

		    for i in range(len(tup)):
		    	reward=tup[i][2]
		    	new_state=tup[i][1]
		    	ts_prob=tup[i][0]
		    	# next_state.append(new_state)
		    	new_V+=(ts_prob* (reward+(gamma*value_func[new_state])))


		    diff=np.abs(new_V-value_func[s])

		    delta=max(delta,diff)
		    value_func[s]=new_V
    
		# print("value_func: ",value_func)
		if delta<tol:
		    it=t
		    break	
	    
	return (value_func,t)




	


def improve_policy(
		env: gym.Env, 
		gamma: float, 
		value_func: np.ndarray, 
		policy: np.ndarray
		) -> np.ndarray:
	"""Performs one step of policy improvement.
	
	Args:
		env: Gym environment
		gamma: Discount factor
		value_func: Value function indexed by state
		policy: Policy function indexed by state 
	Returns:
		Updated policy
	"""


	policy_stable=True
	for s in range(env.nS):
		old_action=policy[s]
		rhs_big=[]
		
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


		# print("RHS_big:",rhs_big)
		policy[s]=np.argmax(np.array(rhs_big))
		if old_action!=policy[s]:
		    policy_stable=False
		
	return policy,policy_stable






def policy_iteration(
		env: gym.Env, 
		gamma: float, 
		max_iterations: int = int(1e3), 
		tol: float = 1e-3) -> Tuple[np.ndarray, np.ndarray, int, int]:
	"""Runs policy iteration until the policy converges.
	
	Args:
		env: Gym enviornment
		gamma: Discount factor
		max_iterations: Max number of iterations before stopping
		tol: Tolerance for convergence
		
	Returns:
		Learned policy function (1D np.ndarray of shape (48) containing the policy at each state)
		Learned value function (1D np.ndarray of shape (48) containing the estimated value at each state)
		Number of policy iterations (int containing the number of iterations until policy iteration converged or returned)
		Total number of policy evaluations (int containing the number of policy evaluation steps)
	"""
	# policy = np.zeros(env.nS, dtype="int")
	# value_func = np.zeros(env.nS)

	policy = np.zeros(env.nS, dtype="int")
	value_func = np.zeros(env.nS)
	# policy_stable=True
	# evaluate_policy(env,value_func,gamma,policy,max_iterations,tol)


	# policy_stable=False
	count=0
	for i in range(max_iterations):
	    # print(i)
	    value_func,it=evaluate_policy(env,value_func,gamma,policy,max_iterations,tol)
	    # print("value_func: ",value_func)
	    count+=it
	    
	    policy,policy_stable=improve_policy(env,gamma,value_func,policy)
	    # print('policy stable:', policy_stable)
	    if policy_stable==True:
	    	break
	    



	
	return (policy,value_func,i,count)   

