import gym
import gym_kuiper_escape
import pygame
from stable_baselines3 import DQN
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import yaml

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

# Parse parameters
model_params = params['model_params']['dqn']

# Parameters
env_id = 'kuiper-escape-base-v0'
n_cpu = 4

# Create environment
env = gym.make(env_id)  # single cpu
# env = make_vec_env(env_id, n_envs=n_cpu)  # multiple cpus

# Create model
model = DQN('MlpPolicy', env, verbose=1, **model_params)

