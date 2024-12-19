import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.dqn import DQNAgent
import random
import numpy as np
from collections import deque

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super().__init__(in_features, out_features, bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.init_parameters()

    def init_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
        
    def forward(self, x):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        if self.training:
            return F.linear(x, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)
        else:
            return F.linear(x, self.weight, bias)

class Noise_DQN(nn.Module):
    def __init__(self, input_dim, output_dim, sigma=0.017):
        super(Noise_DQN, self).__init__()
        self.fc1 = NoisyLinear(input_dim, 64, sigma_init=sigma)
        self.fc2 = NoisyLinear(64, 64, sigma_init=sigma)
        self.fc3 = NoisyLinear(64, output_dim, sigma_init=sigma)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class NoiseDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, gamma=0.9, lr=0.001, epsilon=0.0, epsilon_min=0.0, epsilon_decay=0.995, sigma=0.017, device='cpu'):
        super().__init__(state_dim, action_dim, gamma, lr, epsilon, epsilon_min, epsilon_decay, device)
        self.q_network = Noise_DQN(state_dim, action_dim, sigma=sigma).to(self.device)
        self.target_q_network = Noise_DQN(state_dim, action_dim, sigma=sigma).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_q_network(next_states).max(1, keepdim=True)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()