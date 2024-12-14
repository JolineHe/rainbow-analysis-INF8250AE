import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.ddqn import DoubleDQNAgent
from agents.dqn import DQNAgent
from agents.dueling_ddqn import DuelingDDQNAgent
from agents.multistep_dqn import MultiStepDQNAgent
from agents.prioritized_ddqn import PrioritizedDoubleDQNAgent
from agents.distributional_dqn import DistributionalDQNAgent
from agents.noisy_dqn import NoiseDQNAgent

from arguments import args
from envs.cart_pole import CartPoleEnv

def train_agent(agent, num_episodes=1000):
    max_steps = 50
    batch_size = 32
    returns = []
    for episode in range(num_episodes):
        state_begin, _ = env.reset()
        state, _ = state_begin
        total_reward = 0
        for step in range(max_steps):
            action_idx = agent.act(state)
            action = env.action_space[action_idx]

            next_state, reward, done, _ = env.step(action)

            agent.store_experience(state, action_idx, reward, next_state, done)
            agent.train(batch_size)

            # state_one_hot = next_state_one_hot
            state = next_state
            total_reward += reward

            if done:
                break

        agent.update_target_network()
        agent.decay_epsilon()
        returns.append(total_reward)

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    plt.figure(figsize=(16, 9))
    plt.tight_layout()
    plt.plot(returns)
    plt.title(f"Training Returns of {args.agent} ")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.savefig(f"figures/{args.agent}-cartpole.png")
    plt.show()
    
    # Save the agent parameters
    torch.save(agent.q_network.state_dict(), f"models/{args.agent}_model-cartpole.pth")
    print(f"Model parameters saved to {args.agent}_model.pth")

def test_agent(agent, test_start_state=(2,0), epsilon=0.1):
    model_file = f"models/{args.agent}_model.pth"
    if not os.path.exists(model_file):
        print(f"Error: Model file {model_file} not found. Please train the agent first.")
        return

    # Load the agent parameters
    agent.q_network.load_state_dict(torch.load(model_file))
    agent.q_network.eval()
    print(f"Model parameters loaded from {model_file}")

    agent.epsilon = epsilon
    state = test_start_state
    print(f"start from state:{state+(np.array([1,1]))}")
    env.set_state(state)
    total_reward = 0
    while True:
        env.render()
        state_one_hot = np.zeros(state_dim)
        state_index = state[0] * args.env_size[1] + state[1]
        state_one_hot[state_index] = 1
        action_idx = agent.act(state_one_hot)
        action = env.action_space[action_idx]
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        print(f"Action: {action}, next State: {next_state+(np.array([1,1]))}, Reward: {reward},total reward:{total_reward}, Done: {done}")
        if done:
            break
        state = next_state

if __name__=='__main__':
    env = CartPoleEnv()
    env.render()
    state_dim = env.num_states
    action_dim = len(env.action_space)

    # Choose the agent
    if args.agent == "dqn":
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    elif args.agent == "ddqn":
        agent = DoubleDQNAgent(state_dim=state_dim, action_dim=action_dim)
    elif args.agent == "dueling_ddqn":
        agent = DuelingDDQNAgent(state_dim=state_dim, action_dim=action_dim)
    elif args.agent == "prioritized_ddqn":
        agent = PrioritizedDoubleDQNAgent(state_dim=state_dim, action_dim=action_dim)
    elif args.agent == "distributional_dqn":
        agent = DistributionalDQNAgent(state_dim=state_dim, action_dim=action_dim)
    elif args.agent == "noise_dqn":
        agent = NoiseDQNAgent(state_dim=state_dim, action_dim=action_dim)
    elif args.agent == "multistep_dqn":
        agent = MultiStepDQNAgent(state_dim=state_dim, action_dim=action_dim, n_step=3)

    # Execute based on parameters
    if args.train:
        train_agent(agent, num_episodes=500)

    if args.test:
        test_agent(agent, test_start_state=(7,0), epsilon=0.1)
    

    # Save the trained model parameters
    # torch.save(agent.q_network.state_dict(), 'dqn_model.pth')
    # print("Model parameters saved to dqn_model.pth")


    

    