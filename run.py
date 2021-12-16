
"""
The purpose of this script is to load a previously saved RL agent and run it
through an evaluation sequence (rendering optional).
"""

# 3rd party imports
import gym
import gym_kuiper_escape

# from code.evaluation import evaluate_policy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import A2C

# Create environment instance(s)
env_eval = gym.make('kuiper-escape-base-v0')

# Load and evaluate saved agent
agent = A2C.load('./logs/agent_best/best_model.zip')
evaluate_policy(agent, env_eval, n_eval_episodes=1, render=True)
