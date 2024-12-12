import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from envs.grid_world import GridWorld
from arguments import args
import matplotlib.pyplot as plt
from agents.dqn import DQNAgent


class MultiStepDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, gamma=0.9, lr=0.001, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, n_step=3):
        super().__init__(state_dim, action_dim, gamma, lr, epsilon, epsilon_min, epsilon_decay)
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in n-step buffer and replay buffer.
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            n_step_state, n_step_action, n_step_return, n_step_next_state, n_step_done = self._get_n_step_info()
            super().store_experience(n_step_state, n_step_action, n_step_return, n_step_next_state, n_step_done)

    def _get_n_step_info(self):
        """
        Compute n-step return and get the n-step transition.
        """
        R = 0
        for i, (_, _, reward, _, _) in enumerate(self.n_step_buffer):
            R += (self.gamma ** i) * reward

        n_step_state, n_step_action, _, _, _ = self.n_step_buffer[0]
        _, _, _, n_step_next_state, n_step_done = self.n_step_buffer[-1]

        return n_step_state, n_step_action, R, n_step_next_state, n_step_done

    def train(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_q_network(next_states).max(1, keepdim=True)[0]
        target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__=='__main__':
    # Integrate DQN into GridWorld
    env = GridWorld(env_size=args.env_size,
                    start_state=args.start_state,
                    target_state=args.target_state,
                    forbidden_states=args.forbidden_states)
    # env.reset()
    # env.render()
    # input("stop")


    state_dim = env.num_states
    action_dim = len(env.action_space)

    agent = MultiStepDQNAgent(state_dim=state_dim, action_dim=action_dim, n_step=3)

    # DQN Training loop
    num_episodes = 500
    max_steps = 50
    batch_size = 32
    returns = []
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

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    # Save the trained model parameters
    # torch.save(agent.q_network.state_dict(), 'dqn_model.pth')
    # print("Model parameters saved to dqn_model.pth")


    print("Training completed.")
    plt.figure(figsize=(4,3))
    plt.tight_layout()
    plt.plot(returns)
    plt.show()

    input("press Enter to continue")

    agent.epsilon = 0.1
    state = (2,0)
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


    input('press Enter to quit')