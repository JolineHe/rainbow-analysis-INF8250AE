import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.ddqn import DoubleDQNAgent

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Feature layers
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        # Dueling streams (Noisy)
        self.value_stream = nn.Linear(64, 1)        # value distribution
        self.advantage_stream = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        value = self.value_stream(x)             # [B,1]
        advantage = self.advantage_stream(x)     # [B,action_dim]

        # Dueling combination: Q = V + (A - mean(A))
        advantage_mean = advantage.mean(dim=-1, keepdim=True)   # [B,1]
        q_value = value + (advantage - advantage_mean)          # [B,action_dim]
        return q_value

class DuelingDDQNAgent(DoubleDQNAgent):
    def __init__(self, state_dim, action_dim, gamma=0.9, lr=0.001, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, device='cpu'):
        super().__init__(state_dim, action_dim, gamma, lr, epsilon, epsilon_min, epsilon_decay, device)

        self.q_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_q_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()