import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from agents.my_noisy_dqn import NoisyLinear
from agents.my_prioritized_ddqn import SumTree, PrioritizedReplayBuffer

############################################
# Key Components for Rainbow:
# 1. Double DQN: Use online network to choose actions, target network to evaluate them.
# 2. Dueling Network: Separate value and advantage streams.
# 3. Distributional: Predict a categorical distribution over returns (C51).
# 4. Noisy Networks: NoisyLinear layers instead of epsilon-greedy exploration.
# 5. Prioritized Replay Buffer: Sample more from high-error transitions.
# 6. Multi-step: TD(n) learning for n-step returns.
############################################

class RainbowNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_atoms=51, v_min=0.0, v_max=50.0, sigma_init=0.017):
        super(RainbowNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        # self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)

        # Feature layers
        self.fc1 = NoisyLinear(state_dim, 64, sigma_init)
        self.fc2 = NoisyLinear(64, 64, sigma_init)

        # Dueling streams (Noisy)
        self.value_stream = NoisyLinear(64, self.num_atoms, sigma_init)        # value distribution
        self.advantage_stream = NoisyLinear(64, action_dim * self.num_atoms, sigma_init)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        value = self.value_stream(x).view(-1, 1, self.num_atoms)   # [B,1,num_atoms]
        advantage = self.advantage_stream(x).view(-1, self.action_dim, self.num_atoms) # [B,action_dim,num_atoms]

        # Dueling combination: Q = V + (A - mean(A))
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_dist = value + (advantage - advantage_mean)  # [B, action_dim, num_atoms]
        q_dist = torch.softmax(q_dist, dim=2)          # distributional output
        return q_dist
    
class RainbowAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-4,
                 num_atoms=51, v_min=0.0, v_max=50.0, n_step=4,
                 buffer_size=10000, batch_size=32, alpha=0.6, beta_start=0.4, beta_frames=1000,
                 epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0, sigma=0.017, device='cpu'):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, num_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.batch_size = batch_size
        self.sigma = sigma

        self.q_network = RainbowNetwork(state_dim, action_dim, num_atoms, v_min, v_max, sigma).to(self.device)
        self.target_q_network = RainbowNetwork(state_dim, action_dim, num_atoms, v_min, v_max, sigma).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        # self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha, beta_start, beta_frames)
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)

        # Epsilon not really needed (NoisyNet provides exploration), but keep for compatibility
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def act(self, state):
        # With NoisyNet we don't need epsilon-greedy, just pick max Q-value action.
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.q_network(state_t) # [1, action_dim, num_atoms]
        q_values = (dist * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2) # [1, action_dim]
        return torch.argmax(q_values, dim=1).item()

    def store_to_replay_buffer(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.tensor([action], dtype=int).to(self.device)

        transition = (state, action, reward, next_state, done)

        with torch.no_grad():
            dist = self.q_network(state_tensor) # [1, action_dim, num_atoms]
            dist = dist.gather(1, action_tensor.unsqueeze(-1).expand(1,1,self.num_atoms)).squeeze(1) # [1, num_atoms]

            online_next_dist = self.q_network(next_state_tensor) # [1, action_dim, num_atoms]
            next_q = (online_next_dist * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2) # [1, action_dim]
            next_action = torch.argmax(next_q, dim=1, keepdim=True) # [1,1]

            target_next_dist = self.target_q_network(next_state_tensor) # [1, action_dim, num_atoms]
            target_next_dist = target_next_dist.gather(1, next_action.unsqueeze(-1).expand(1,1,self.num_atoms)).squeeze(1) # [1, num_atoms]

            target_next_dist = target_next_dist # [1, num_atoms]
            reward = torch.tensor([reward]).to(self.device)
            done = torch.tensor([done], dtype=int).to(self.device)

            projected_dist = self.project_distribution_vec(target_next_dist, reward, done)
        
            dist_log = torch.log(dist + 1e-8)
            loss_element = -(projected_dist * dist_log).sum(dim=1)

        error = torch.abs(loss_element).detach().cpu().numpy()
        self.replay_buffer.push(error, transition)


    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in n-step buffer and replay buffer.
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step or done:
            n_step_state, n_step_action, n_step_return, n_step_next_state, n_step_done = self._get_n_step_info()
            self.store_to_replay_buffer(n_step_state, n_step_action, n_step_return, n_step_next_state, n_step_done)
            
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
    
    def project_distribution_vec(self, next_dist, rewards, dones):
        num_atoms = self.support.size(0)
        delta_z = (self.v_max - self.v_min) / (num_atoms - 1)

        # Compute Tz and clamp to range [v_min, v_max]
        Tz = rewards + (1 - dones) * self.gamma * self.support[None, :]
        Tz = torch.clamp(Tz, self.v_min, self.v_max)

        # Compute projection indices
        b = (Tz - self.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # Clamp indices to valid range
        l = torch.clamp(l, 0, num_atoms - 1)
        u = torch.clamp(u, 0, num_atoms - 1)

        # Compute offsets for interpolation
        u_offset = (u.float() - b)
        l_offset = (b - l.float())

        # Handle edge case where l == u
        eq_mask = (l == u)
        l_offset[eq_mask] = 1.0  # Assign the full probability to `l` if `l == u`
        u_offset[eq_mask] = 0.0

        # Initialize the projected distribution
        projected_dist = torch.zeros_like(next_dist)

        # Scatter-add probabilities to the projected distribution
        projected_dist.scatter_add_(
            1, l, (next_dist * u_offset).view_as(l)
        )
        projected_dist.scatter_add_(
            1, u, (next_dist * l_offset).view_as(u)
        )

        return projected_dist
    
    def train(self, batch_size=None):
        if len(self.replay_buffer) < self.batch_size:
            return
        if batch_size is None:
            batch_size = self.batch_size

        batch, tree_idx, is_weights = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = map(np.array, batch)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(-1).to(self.device)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current dist
        dist = self.q_network(states) # [B, action_dim, num_atoms]
        dist = dist.gather(1, actions.unsqueeze(-1).expand(batch_size,1,self.num_atoms)).squeeze(1) # [B, num_atoms]

        with torch.no_grad():
            # Double DQN: choose best action via q_network, eval via target_q_network
            online_next_dist = self.q_network(next_states) # [B, action_dim, num_atoms]
            next_q = (online_next_dist * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2) # [B, action_dim]
            next_actions = torch.argmax(next_q, dim=1, keepdim=True) # [B,1]

            target_next_dist = self.target_q_network(next_states) # [B, action_dim, num_atoms]
            target_next_dist = target_next_dist.gather(1, next_actions.unsqueeze(-1).expand(batch_size,1,self.num_atoms)).squeeze(1) # [B, num_atoms]

            projected_dist = self.project_distribution_vec(target_next_dist, rewards, dones)

        dist_log = torch.log(dist + 1e-8)
        loss_element = -(projected_dist * dist_log).sum(dim=1)
        errors = torch.abs(loss_element).detach().cpu().numpy()
        loss = (loss_element * is_weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        errors = loss_element.detach().cpu().numpy()
        self.replay_buffer.batch_update(tree_idx, errors)


    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        # Not needed for NoisyNet, but keep for compatibility
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)