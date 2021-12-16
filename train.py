"""
The purpose of this script is to train an AI agent to play the custom-built
Kuiper Escape game using the A2C reinforcement learning algorithm.
"""

# 3rd party imports
import gym
import gym_kuiper_escape

# from code.evaluation import evaluate_policy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import EvalCallback
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
