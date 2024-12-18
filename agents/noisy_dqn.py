import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.dqn import DQNAgent
import random
import numpy as np
# from collections import deque

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        # self.weight_sigma = torch.ones((out_features, in_features), requires_grad=False) * self.sigma_init
        # self.bias_sigma = torch.ones(out_features, requires_grad=False) * self.sigma_init

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        # mu_range = 1
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            # weight = self.weight_mu
            # bias = self.bias_mu
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

class Noise_DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Noise_DQN, self).__init__()
        self.fc1 = NoisyLinear(input_dim, 64)
        self.fc2 = NoisyLinear(64, 64)
        self.fc3 = NoisyLinear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class NoiseDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, gamma=0.9, lr=0.001, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, device='cpu'):
        super().__init__(state_dim, action_dim, gamma, lr, epsilon, epsilon_min, epsilon_decay, device)
        self.q_network = Noise_DQN(state_dim, action_dim).to(self.device)
        self.target_q_network = Noise_DQN(state_dim, action_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        # self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        # if np.random.rand() < self.epsilon:
        #     return np.random.randint(self.action_dim)
        self.reset_noise()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def reset_noise(self):
        self.q_network.fc1.reset_noise()
        self.q_network.fc2.reset_noise()
        self.q_network.fc3.reset_noise()
        self.target_q_network.fc1.reset_noise()
        self.target_q_network.fc2.reset_noise()
        self.target_q_network.fc3.reset_noise()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        # self.reset_noise()

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

        self.reset_noise()
        q_values = self.q_network(states).gather(1, actions)

        # no grad instead of value cuz we need to evaluate with noise
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # print("Mu grad norm: ", self.q_network.fc1.weight_mu.grad.norm().item())
        # print("Sigma grad norm: ", self.q_network.fc1.weight_sigma.grad.norm().item())
        self.optimizer.step()
