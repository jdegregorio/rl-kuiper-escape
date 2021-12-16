# Reinforcement Learning to Train PyGame Agent

## Overview 

The purpose of this project is to demonstrate how reinforcement learning can be used to train an AI Agent/Player in a custom home-built python game

## Demo - Trained Agent (A2C)

`TODO: Insert GIF`

## Custom Gym Environment

### Game Overview - Kuiper Escape

The objective of the game is to live as long as you can, while avoiding the asteroids in the Kuiper Belt. The player starts centered in the screen, an can move in any direction using the keyboard arrows.  Asteroids are generated with random sizes, speeds, and starting locations. The game ends once all player lives have expired.

Check out the repo for the custom Open AI Gym environment using PyGame framework :point_right:  [REPO](https://github.com/jdegregorio/gym-kuiper-escape)

<img width="405" alt="image" src="https://user-images.githubusercontent.com/20359930/144731391-99aa8834-6744-48e8-8a18-8ea3e0c8d2af.png">

### Actions 

The user has the following discrete actions:
 * 0: Don't move
 * 1: Up
 * 2: Right
 * 3: Down
 * 4: Left
 * 5: Up/Right Diagnal
 * 6: Right/Down Diagnal
 * 7: Down/Left Diagnal
 * 8: Left/Up Diagnal

### State/Observations

The state/observation is a "virtual" lidar system. It sends off virtual
beams of light in all directions to gather an array of points describing
the distance and characteristics of nearby objects. The size of the lidar array and resulting observation/state space is configurable when the environment is initialized

The observation data (for each beam in the lidar array):
 * Distance (i.e. radial distance from player to terminating point of lidar beam)
 * Collision detection
   * 0 if terminated at edge of screen, or at max radius distance
   * 1 if collided with a rock

**Example Visualizations of State**

<img width="264" alt="image" src="https://user-images.githubusercontent.com/20359930/146223524-e07f7dd8-7e5e-40e2-a374-fdb20f987153.png">
<img width="261" alt="image" src="https://user-images.githubusercontent.com/20359930/146223615-de23593f-02df-4ef1-b356-87153208d6f1.png">

Note: The yellow dots (1 collide state) represent contact with a rock, the green dots (0 collide state) represent contact with wall or open space.

### Rewards

The environment will provide the following rewards:
 * Reward of 1 for each step without losing life
 * No reward is given if the player is in the corners of the screen

## Setup

### Installation
First it is recommended to setup a virtual environment:
```bash
python -m venv .env
source .env/bin/activate
```

Update pip and wheel to ensure a smooth installation process:
```bash
pip install --upgrade pip
pip install --upgrade wheel
```

Finally, install package locally with pip:
```bash
git clone https://github.com/jdegregorio/gym-kuiper-escape.git
pip install -e gym-kuiper-escape
```

To uninstall, use the following:
```bash
pip uninstall gym-kuiper-escape
```

### Development Environment Details
 * Python Version: 3.8.10
 * Operating System: Ubuntu 20.04.3 LTS

## Background & Resources

### Open AI Gym Framework
Open AI Gym provides a standardized framework for training reinforcement learning models. The framework has numerous built-in environments (often games) for experimentation, but also enables users to define their own custom environments.

 * [Open AI Gym Documentation](https://gym.openai.com/docs/)
 * [Creating Customer Environments](https://github.com/openai/gym/blob/master/docs/creating_environments.md)
 * [Example Custom Environment](https://github.com/openai/gym-soccer/blob/master/gym_soccer/envs/soccer_env.py)
 * [Core Open AI Gym Clases](https://github.com/openai/gym/blob/master/gym/core.py)

### PyGame Framework

PyGame is a framework for developing games within python. 

This [tutorial](https://realpython.com/pygame-a-primer/) is a great primer for getting started.

### Stable Baselines

Stable baselines was the primary reinforcement learning framework used for this project. It provides a great set of algorithms and monitoring tools to get started. Future work on this project will explore other packages such as Acme.
