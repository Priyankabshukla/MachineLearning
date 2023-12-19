import argparse

import gymnasium as gym

import agents
import envs
import utils



def run_env(env: gym.Env, agent: agents.Agent, gamma: int):
	agent.learn(gamma=gamma)

	if isinstance(agent, agents.PolicyAgent):
		utils.visualize_value(env, agent.value_func)
		utils.visualize_policy(env, agent.policy)
	
	if isinstance(agent, agents.ValueAgent):
		utils.visualize_value(env, agent.value_func)

	observation, info = env.reset(seed=42)
	for _ in range(1000):
		action = agent.sample(observation) 
		observation, reward, terminated, truncated, info = env.step(action)

		if terminated or truncated:
			env.close()
			return 


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--env", choices=["default", "stochastic", "shaped_reward"], 
		default="default")
	parser.add_argument(
		"--agent", choices=["random", "policy", "value"], default="random")
	parser.add_argument(
		"--render_mode", choices=["human", "rgb_array"], default=None)
	parser.add_argument("--gamma", type=float, default=0.99)
		
	args = parser.parse_args()

	if args.env == "default":
		env = envs.ModifiedCliffWalkingEnv(render_mode=args.render_mode) 
	elif args.env == "stochastic":
		env = envs.StochasticCliffWalkingEnv(render_mode=args.render_mode)
	elif args.env == "shaped_reward":
		env = envs.ShapedRewardCliffWalkingEnv(render_mode=args.render_mode)
	
	if args.agent == "random":
		agent = agents.RandomAgent(env)
	elif args.agent == "policy":
		agent = agents.PolicyAgent(env)
	elif args.agent == "value":
		agent = agents.ValueAgent(env)

	run_env(env, agent, args.gamma)

