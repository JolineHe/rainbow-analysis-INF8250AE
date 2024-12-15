import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.dqn import DQNAgent


class Distributional_DQN(nn.Module):
    def __init__(self, input_dim, action_dim, num_atoms=51, v_min=-10.0, v_max=10.0):
        super(Distributional_DQN, self).__init__()
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim * num_atoms)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.action_dim, self.num_atoms)
        x = F.softmax(x, dim=2)
        return x

def project_distribution(next_dist, rewards, dones, gamma, support, v_min, v_max):
    batch_size = rewards.size(0)
    num_atoms = support.size(0)
    delta_z = (v_max - v_min) / (num_atoms - 1)

    projected_dist = torch.zeros_like(next_dist)  # [batch_size, num_atoms]

    batch_indices = torch.arange(batch_size)

    for i in range(num_atoms):
        Tz = rewards + (1 - dones) * gamma * support[i]
        Tz = torch.clamp(Tz, v_min, v_max)

        b = (Tz - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l = torch.clamp(l, 0, num_atoms - 1)
        u = torch.clamp(u, 0, num_atoms - 1)

        eq_mask = (l == u)
        not_eq_mask = ~eq_mask

        dist_slice = next_dist[:, i]

        if eq_mask.any():
            eq_idx = eq_mask.nonzero(as_tuple=True)[0]  # eq_idx为eq_mask为True的batch下标
            l_eq = l[eq_idx]
            projected_dist[eq_idx, l_eq] += dist_slice[eq_idx]

        if not_eq_mask.any():
            neq_idx = not_eq_mask.nonzero(as_tuple=True)[0]
            l_neq = l[neq_idx]
            u_neq = u[neq_idx]

            u_offset = (u.float() - b)[neq_idx]
            l_offset = (b - l.float())[neq_idx]

            projected_dist[neq_idx, l_neq] += dist_slice[neq_idx] * u_offset
            projected_dist[neq_idx, u_neq] += dist_slice[neq_idx] * l_offset

    return projected_dist

class DistributionalDQNAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.0001,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 num_atoms=51, v_min=-10.0, v_max=10.0):
        super().__init__(state_dim, action_dim, gamma, lr, epsilon, epsilon_min, epsilon_decay)
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        self.q_network = Distributional_DQN(state_dim, action_dim, num_atoms, v_min, v_max)
        self.target_q_network = Distributional_DQN(state_dim, action_dim, num_atoms, v_min, v_max)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist = self.q_network(state_tensor)  # [1, action_dim, num_atoms]
        q_values = torch.sum(dist * self.support.unsqueeze(0).unsqueeze(0), dim=2)
        return torch.argmax(q_values, dim=1).item()

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

        dist = self.q_network(states)
        dist = dist.gather(1, actions.unsqueeze(-1).expand(batch_size, 1, self.num_atoms))  # [batch,1,num_atoms]
        dist = dist.squeeze(1)  # [batch, num_atoms]

        with torch.no_grad():
            next_dist = self.target_q_network(next_states)  # [batch, action_dim, num_atoms]
            online_next_dist = self.q_network(next_states)  # [batch, action_dim, num_atoms]
            next_q = (online_next_dist * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # [batch, action_dim]
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)  # [batch, 1]

            next_dist = next_dist.gather(1, next_actions.unsqueeze(-1).expand(batch_size, 1, self.num_atoms))
            next_dist = next_dist.squeeze(1)  # [batch, num_atoms]

            projected_dist = project_distribution(next_dist, rewards, dones, self.gamma,
                                                  self.support, self.v_min, self.v_max)  # [batch, num_atoms]

        dist_log = torch.log(dist + 1e-8)
        loss = - (projected_dist * dist_log).sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.decay_epsilon()

    