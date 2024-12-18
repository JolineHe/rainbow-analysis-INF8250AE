from envs.grid_world import GridWorld
from arguments import args
import matplotlib.pyplot as plt
from agents.dqn import DQNAgent
from agents.ddqn import DoubleDQNAgent
from agents.dueling_ddqn import DuelingDDQNAgent
from agents.multistep_dqn import MultiStepDQNAgent
from agents.prioritized_ddqn import PrioritizedDoubleDQNAgent
from agents.distributional_dqn import DistributionalDQNAgent
from agents.a3c_dqn import A3CAgent
from agents.noisy_dqn import NoiseDQNAgent
from agents.rainbow import RainbowAgent
import numpy as np
import torch
import os
import pickle
import pandas as pd

def train_agent(agent, num_episodes=1000):
    max_steps = 50
    batch_size = 32
    returns = []
    steps = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state_one_hot = np.zeros(state_dim)
        state_index = state[0] * args.env_size[1] + state[1]
        state_one_hot[state_index] = 1

        total_reward = 0
        for step in range(max_steps):
            action_idx = agent.act(state_one_hot)
            action = env.action_space[action_idx]

            next_state, reward, done, _ = env.step(action)
            next_state_one_hot = np.zeros(state_dim)
            next_state_index = next_state[0] * args.env_size[1] + next_state[1]
            next_state_one_hot[next_state_index] = 1

            agent.store_experience(state_one_hot, action_idx, reward, next_state_one_hot, done)
            agent.train(batch_size)

            state_one_hot = next_state_one_hot
            total_reward += reward
            if done:
                break

        agent.update_target_network()
        agent.decay_epsilon()
        returns.append(total_reward)
        steps.append(episode)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    return steps, returns
    # plt.figure(figsize=(16, 9))
    # plt.tight_layout()
    # plt.plot(returns)
    # plt.title(f"Training Returns of {args.agent} ")
    # plt.xlabel("Episodes")
    # plt.ylabel("Total Reward")
    # plt.savefig(f"figures/{args.agent}.png")
    # plt.show()
    #
    # # Save the agent parameters
    # torch.save(agent.q_network.state_dict(), f"models/{args.agent}_model.pth")
    # print(f"Model parameters saved to {args.agent}_model.pth")


def test_agent(agent, test_start_state=(2,0), epsilon=0.1, max_step=200):
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
    step = 0
    while True:
        env.render()
        step += 1
        state_one_hot = np.zeros(state_dim)
        state_index = state[0] * args.env_size[1] + state[1]
        state_one_hot[state_index] = 1
        action_idx = agent.act(state_one_hot)
        action = env.action_space[action_idx]
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        print(f"step:{step}, Action: {action}, next State: {next_state+(np.array([1,1]))}, Reward: {reward},total reward:{total_reward}, Done: {done}")
        if done:
            break
        if step == max_step:
            print(f"stop because cann't reach the goal within {step} steps")
            break
        state = next_state

if __name__=='__main_single__':
    # Integrate DQN into GridWorld
    env = GridWorld(env_size=args.env_size,
                    start_state=args.start_state,
                    target_state=args.target_state,
                    forbidden_states=args.forbidden_states)
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
    elif args.agent == "a3c":
        agent = A3CAgent(state_dim=state_dim, action_dim=action_dim)
    elif args.agent == "rainbow":
        agent = RainbowAgent(state_dim=state_dim, action_dim=action_dim)
    elif args.agent == "multistep_dqn":
        agent = MultiStepDQNAgent(state_dim=state_dim, action_dim=action_dim, n_step=4)

    # Execute based on parameters
    if args.train:
        train_agent(agent, num_episodes=1000)

    if args.test:
        test_agent(agent, test_start_state=(1,1), epsilon=0.3, max_step=200)
        input("press Enter to quit")

    # Save the trained model parameters
    # torch.save(agent.q_network.state_dict(), 'dqn_model.pth')
    # print("Model parameters saved to dqn_model.pth")

if __name__ == '__main__':
    env_size = args.env_size
    env = GridWorld(env_size=env_size,
                    start_state=args.start_state,
                    target_state=args.target_state,
                    forbidden_states=args.forbidden_states)
    # env.render()
    state_dim = env.num_states
    action_dim = len(env.action_space)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    agents = ['dqn', 'ddqn', 'multistep_dqn', 'dueling_ddqn', 'distributional_dqn', 'noise_dqn', 'a3c', 'rainbow', 'prioritized_ddqn']
    # agents = ['dqn']
    results = {}
    for agent_name in agents:
        print(agent_name)
        args.agent = agent_name
        seeds_returns = {}
        # for SEED in (range(40, 51)):
        for SEED in (range(40, 51)):
            print(SEED)
            np.random.seed(SEED)
            if agent_name == "dqn":
                agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
            elif agent_name == "ddqn":
                agent = DoubleDQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
            elif agent_name == "dueling_ddqn":
                agent = DuelingDDQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
            elif agent_name == "distributional_dqn":
                agent = DistributionalDQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
            elif agent_name == "noise_dqn":
                agent = NoiseDQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
            elif agent_name == "a3c":
                agent = A3CAgent(state_dim=state_dim, action_dim=action_dim, device=device)
            elif agent_name == "rainbow":
                agent = RainbowAgent(state_dim=state_dim, action_dim=action_dim, device=device)
            elif agent_name == "multistep_dqn":
                agent = MultiStepDQNAgent(state_dim=state_dim, action_dim=action_dim, n_step=3, device=device)

            # Execute based on parameters
            if args.train:
                steps, returns = train_agent(agent, num_episodes=500)
                value_function = np.zeros(env_size)
                for x in range(env_size[0]):
                    for y in range(env_size[1]):
                        state = (x, y)
                        state_one_hot = np.zeros(state_dim)
                        state_index = state[0] * args.env_size[1] + state[1]
                        state_one_hot[state_index] = 1
                        state_tensor = torch.FloatTensor(state_one_hot).unsqueeze(0).to(agent.device)
                        with torch.no_grad():
                            q_values = agent.q_network(state_tensor)
                        if type(q_values) == tuple:
                            q_values, t = q_values
                        value_function[state] = q_values.max().item()
                seeds_returns[SEED] = (steps, returns, value_function)

            if args.test:
                test_agent(agent, test_start_state=(7, 0), epsilon=0.1)

        plt.figure(figsize=(16, 9))
        plt.tight_layout()
        mean_return = np.stack([seeds_returns[k][1] for k in seeds_returns.keys()]).mean(0)
        std_return = np.stack([seeds_returns[k][1] for k in seeds_returns.keys()]).std(0)
        mean_value_function = np.stack([seeds_returns[k][2] for k in seeds_returns.keys()]).mean(0)
        std_value_function = np.stack([seeds_returns[k][2] for k in seeds_returns.keys()]).std(0)
        mean_value_df = pd.DataFrame(mean_value_function, columns=[f'Y{y}' for y in range(env_size[1])],
                                    index=[f'X{x}' for x in range(env_size[0])])
        std_value_df = pd.DataFrame(std_value_function, columns=[f'Y{y}' for y in range(env_size[1])],
                                    index=[f'X{x}' for x in range(env_size[0])])

        mean_value_df.to_csv(f"figures/{args.agent}_mean_value_function.csv")
        std_value_df.to_csv(f"figures/{args.agent}_std_value_function.csv")

        results[agent_name] = (mean_return, std_return)
        plt.plot(steps, mean_return, label='dqn')
        plt.title(f"Training Returns of {args.agent} ")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.fill_between(steps, mean_return - std_return, mean_return + std_return, alpha=0.3)
        # plt.ylim(0, 210)
        plt.savefig(f"figures/{args.agent}.png")
        # plt.show()

    # Save the trained model parameters
    # torch.save(agent.q_network.state_dict(), 'dqn_model.pth')
    # print("Model parameters saved to dqn_model.pth")
    with open('results/results_grip.pkl', 'wb') as f:
        pickle.dump(results, f)
    

    