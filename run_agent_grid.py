from envs.grid_world import GridWorld
from arguments import args
import matplotlib.pyplot as plt
import seaborn as sns
from agents.dqn import DQNAgent
from agents.ddqn import DoubleDQNAgent
from agents.my_dueling_ddqn import DuelingDDQNAgent
from agents.multistep_dqn import MultiStepDQNAgent
from agents.my_prioritized_ddqn import PrioritizedDoubleDQNAgent
from agents.distributional_dqn import DistributionalDQNAgent
from agents.a3c_dqn import A3CAgent
from agents.noisy_dqn import NoiseDQNAgent
from agents.my_rainbow import RainbowAgent
import numpy as np
import torch
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image
import io

def train_agent(agent, num_episodes=500):
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

def save_agent(agent, agent_name):
    # Save the agent parameters
    torch.save(agent.q_network.state_dict(), f"results/models/{agent_name}_model.pth")
    print(f"Model parameters saved to results/models/{agent_name}_model.pth")


def cal_value_function(seeds_result):
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
            if agent_name in ["distributional_dqn", "rainbow" ]:
                q_values = torch.sum(q_values * agent.support, dim=2)
            if type(q_values) == tuple:
                q_values, t = q_values
            value_function[state] = q_values.max().item()
    return value_function

def save_value_function(agent_name, seeds_result):  
    mean_value_function = np.stack([seeds_result[k][2] for k in seeds_result.keys()]).mean(0)
    std_value_function = np.stack([seeds_result[k][2] for k in seeds_result.keys()]).std(0)
    mean_value_df = pd.DataFrame(mean_value_function, columns=[f'Y{y}' for y in range(env_size[1])],
                                index=[f'X{x}' for x in range(env_size[0])])
    std_value_df = pd.DataFrame(std_value_function, columns=[f'Y{y}' for y in range(env_size[1])],
                                index=[f'X{x}' for x in range(env_size[0])])

    values = (mean_value_df, std_value_df)
    with open(f'results/value_functions/{agent_name}_grid_value.pkl', 'wb') as f:
        pickle.dump(values, f)  
        print(f"value functions saved to results/value_functions/{agent_name}_grid_value.pkl")



def save_returns(agent_name, seeds_result):
    mean_return = np.stack([seeds_result[k][1] for k in seeds_result.keys()]).mean(0)
    std_return = np.stack([seeds_result[k][1] for k in seeds_result.keys()]).std(0)
    
    results = (mean_return, std_return)
    with open(f'results/training_returns/{agent_name}_grid_return.pkl', 'wb') as f:
        pickle.dump(results, f)
        print(f"training returns saved to results/training_returns/{agent_name}_grid_return.pkl")


def plot_returns(agent_name):
    with open(f'results/training_returns/{agent_name}_grid_return.pkl', 'rb') as f:
        results = pickle.load(f)
    mean_return, std_return = results

    plt.figure(figsize=(16, 9))
    plt.tight_layout()
    plt.plot(mean_return)
    plt.title(f"Training Returns of {agent_name} ")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.fill_between(steps, mean_return - std_return, mean_return + std_return, alpha=0.3)
    # plt.ylim(0, 210)
    plt.savefig(f"results/figures/training_returns/{agent_name}.png")
    print(f'training returns saved to results/figures/training_returns/{agent_name}.png')
    # plt.show()

def plot_heatmap(agent_name):
    with open(f'results/value_functions/{agent_name}_grid_value.pkl', 'rb') as f:
        mean_return, _ = pickle.load(f)
    
    mean_return = mean_return.T

    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_return, annot=True, cmap='coolwarm', cbar=True)

    plt.title(f'Heatmap of {agent_name} Value Function')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.savefig(f"results/figures/value_function/{agent_name}_value_function.png")
    print(f'value function heatmap saved to results/figures/value_function/{agent_name}_heatmap.png')
    # plt.show()


def test_agent(agent, agent_name, test_start_state=(2,0), epsilon=0.1, max_step=200):
    model_file = f"results/models/{agent_name}_model.pth"
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

    frames = []
    while True:
        env.render()
        # Save the current figure
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        frames.append(Image.open(img_buf))

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
    # Save as GIF
    imageio.mimsave(f'results/figures/gif/{agent_name}_grid_test.gif', frames, duration=0.5)
    print(f'gif saved to figures/gif/{agent_name}_grid_test.gif')

def create_agent(agent_name):
    if agent_name == "dqn":
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    elif agent_name == "ddqn":
        agent = DoubleDQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    elif agent_name == "dueling_ddqn":
        agent = DuelingDDQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    elif agent_name == "distributional_dqn":
        agent = DistributionalDQNAgent(state_dim=state_dim, action_dim=action_dim, v_min=-30, v_max=30., epsilon_decay=0.99, device=device)
    elif agent_name == "noise_dqn":
        agent = NoiseDQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    elif agent_name == "a3c":
        agent = A3CAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    elif agent_name == "multistep_dqn":
        agent = MultiStepDQNAgent(state_dim=state_dim, action_dim=action_dim, n_step=3, device=device)
    elif args.agent == "prioritized_ddqn":
        agent = PrioritizedDoubleDQNAgent(state_dim=state_dim, action_dim=action_dim)
    elif agent_name == "rainbow":
        agent = RainbowAgent(state_dim=state_dim, action_dim=action_dim, v_min=-30, v_max=30., n_step=4, sigma=0.1, device=device)
    return agent

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
    if args.agent == None:
        agents = ['dueling_ddqn', 'distributional_dqn', 'noise_dqn', 'prioritized_ddqn', 'rainbow']
        # agents = ['multistep_dqn', 'dueling_ddqn', 'distributional_dqn', 'noise_dqn', 'prioritized_ddqn', 'rainbow']
    else:
        agents = [args.agent]
    results = {}
    print(args.train, args.test)
    for agent_name in agents:
        print(agent_name)
        if args.train:
            seeds_results = {}
            for SEED in (range(40, 50)):
                print(SEED)
                np.random.seed(SEED)
                agent = create_agent(agent_name)
                steps, returns = train_agent(agent, num_episodes=500)
                value_function = cal_value_function(agent)
                seeds_results[SEED] = (steps, returns, value_function)

            save_agent(agent, agent_name)
            save_value_function(agent_name, seeds_results)
            save_returns(agent_name, seeds_results)
            plot_returns(agent_name)
            plot_heatmap(agent_name)

        if args.test:
            agent = create_agent(agent_name)
            test_agent(agent, agent_name, test_start_state=(7, 0), epsilon=0.1)
    

    