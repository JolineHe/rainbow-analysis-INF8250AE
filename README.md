# Deep Reinforcement Learning Implementations

This project implements various deep reinforcement learning algorithms to solve classic control problems including Grid World navigation and CartPole balancing tasks.

## Environments

### Grid World
A customizable grid navigation task where:
- The agent starts at a specified position
- The goal is to reach a target position 
- There are forbidden states (obstacles) to avoid
- Actions are: up, down, left, right
- Rewards: -1 for each step, -30 for hitting obstacles, +30 for reaching goal

### CartPole
The classic CartPole-v1 from OpenAI Gym where:
- The goal is to balance a pole on a moving cart
- Actions are: push cart left or right
- Episode ends if pole angle exceeds ±15° or cart moves out of bounds
- Reward of +1 for each timestep pole remains upright

## Algorithms Implemented

- DQN (Deep Q-Network)
- Double DQN 
- Dueling Double DQN
- Distributional DQN
- Noisy DQN
- Multi-step DQN
- Prioritized Double DQN
- Rainbow DQN

## Usage

### Grid World

1. To train and test a specifique agent :

    python run_agent_grid.py --agent=ddqn --train=True --test=True

1. To train and test a batch of agents :

    python run_agent_grid.py --train=True
