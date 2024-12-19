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
    def __init__(self, state_dim, action_dim, gamma=0.9, lr=0.001, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, n_step=4, device='cpu'):
        super().__init__(state_dim, action_dim, gamma, lr, epsilon, epsilon_min, epsilon_decay, device)
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in n-step buffer and replay buffer.
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step or done:
            n_step_state, n_step_action, n_step_return, n_step_next_state, n_step_done = self._get_n_step_info()
            super().store_experience(n_step_state, n_step_action, n_step_return, n_step_next_state, n_step_done)
    

    def _get_n_step_info(self):
        """
        Compute n-step return and get the n-step transition.
        """
        R = 0
        for i, (_, _, reward, _, done) in enumerate(self.n_step_buffer):
            R += (self.gamma ** i) * reward
            if done:
                break

        n_step_state, n_step_action, _, _, _ = self.n_step_buffer[0]
        _, _, _, n_step_next_state, n_step_done = self.n_step_buffer[i]

        return n_step_state, n_step_action, R, n_step_next_state, n_step_done


    def train(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        # states, actions, rewards, next_states, dones = zip(*batch)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_q_network(next_states).max(1, keepdim=True)[0]
        target_q_values = rewards + (1 - dones) * self.gamma**self.n_step * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


