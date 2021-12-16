# rl-kuiper-escape
A project to  build an AI player for custom game (Kuiper Escape) using reinforcement learning (RL) algorithms.

## Setup


Pre-requisites:
```
pip install tensorflow
```

Installing Baselines
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```


## DQN Parmeters

 * param learning_rate: The learning rate, it can be a function of the current progress remaining (from 1 to 0)
 * param buffer_size: size of the replay buffer
 * param learning_starts: how many steps of the model to collect transitions for before learning starts
 * param batch_size: Minibatch size for each gradient update
 * param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
 * param gamma: the discount factor
 * param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit like ``(5, "step")`` or ``(2, "episode")``. Set to ``-1`` means to do as many gradient steps as steps done in the environment during the rollout.
 * param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``). If ``None``, it will be automatically selected.
 * param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
 * param optimize_memory_usage: Enable a memory efficient variant of the replay buffer at a cost of more complexity.
 * param target_update_interval: update the target network every ``target_update_interval`` environment steps.
 * param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
 * param exploration_initial_eps: initial value of random action probability
 * param exploration_final_eps: final value of random action probability
 * param max_grad_norm: The maximum value for the gradient clipping
 * param tensorboard_log: the log location for tensorboard (if None, no logging) used for evaluating the agent periodically. (Only available when passing string for the environment)
 * param policy_kwargs: additional arguments to be passed to the policy on creation
 * param verbose: the verbosity level: 0 no output, 1 info, 2 debug
 * param seed: Seed for the pseudo random generators
 * param device: Device (cpu, cuda, ...) on which the code should be run. Setting it to auto, the code will be run on the GPU if possible.
 * param _init_setup_model: Whether or not to build the network at the creation of the instance