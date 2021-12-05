# Standard imports
import sys
import os

# 3rd party imports
import numpy as np
import yaml
import gym
import gym_kuiper_escape
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# Local imports
path = os.getcwd() + '/code/'
sys.path.insert(0, path)
from evaluation import evaluate_policy

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

# Parse parameters
model_params = params['model_params']['dqn']

# Parameters
env_id = 'kuiper-escape-base-v0'
n_cpu = 4

# Create environment
env = gym.make(env_id)
env = Monitor(env, './')  # Monitor logs

# Create model
model = DQN('MlpPolicy', env, verbose=1, **model_params)

# Evaluate baseline
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, render=False)

# Test drive dummy model
env = model.get_env()
all_episode_rewards = []
for i in range(10):
    episode_rewards = []
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards.append(reward)
    all_episode_rewards.append(sum(episode_rewards))

mean_episode_reward = np.mean(all_episode_rewards)
print("Mean reward:", mean_episode_reward)