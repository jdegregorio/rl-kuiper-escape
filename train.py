# Standard imports
import sys
import os
import time
import random

# 3rd party imports
import numpy as np
import yaml
import gym
import gym_kuiper_escape
# from code.evaluation import evaluate_policy
from stable_baselines.common.evaluation import evaluate_policy
from code.callback import CustomEvalCallback
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

# Create environment instance(s)
env_eval = gym.make('kuiper-escape-base-v0')
env = make_vec_env('kuiper-escape-base-v0', n_envs=4)

# Define callback function
eval_callback = EvalCallback(
    env_eval, 
    best_model_save_path='./logs/agent_best',
    log_path='./logs/',
    eval_freq=10000,
    deterministic=True, 
    render=False
)

# Create agent model
agent = A2C('MlpPolicy', env, verbose=1, tensorboard_log='./tensorboard/')

# Train agent
agent.learn(5000000, 
    reset_num_timesteps=False,
    callback=eval_callback
)
agent.save("agent")

agent = A2C.load('./logs/agent_best/best_model.zip')
evaluate_policy(agent, env_eval, n_eval_episodes=1, render=True)

env_test = gym.make('kuiper-escape-base-v0')
obs = env_test.reset()
zero_completed_obs = np.zeros((agent.n_envs,) + agent.observation_space.shape)
zero_completed_obs[0, :] = obs
obs = zero_completed_obs
for i in range(100):
    action, _states = agent.predict(obs)
    obs, rewards, dones, info = env_test.step(action[0])
    env_test.render()