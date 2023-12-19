import gymnasium as gym
import numpy as np

def print_table(arr):
	# Assumes arr is of shape (4 x 12).
	table_str = "\\begin{center}\\begin{tabular}{ c c c c c c c c c c c c }"
	for row in arr:
		for elt in row:
			table_str += str(elt) + " & "
		table_str += ' \\\\ '
	table_str += "\\end{tabular}\\end{center}"
	print(table_str)


def visualize_value(env: gym.Env, value_func: np.ndarray):
	vf = np.around(value_func.reshape(env.shape), 2)
	
	print('Value function:')
	print(vf)
	
	print('You may find the follow latex table useful for submission: ')
	print_table(vf)
	print()

def visualize_policy(env: gym.Env, policy: np.ndarray):
	arrows = ['\u2191', '\u2192', '\u2193', '\u2190']
	parr = np.vectorize(lambda action: arrows[action])(policy).reshape(env.shape) 
	
	print('Policy:')
	print(parr)
	
	print('You may find the follow latex table useful for submission: ')
	print_table(parr)
	print()
