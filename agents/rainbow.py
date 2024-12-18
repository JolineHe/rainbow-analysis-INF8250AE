import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


############################################
# Key Components for Rainbow:
# 1. Double DQN: Use online network to choose actions, target network to evaluate them.
# 2. Dueling Network: Separate value and advantage streams.
# 3. Distributional: Predict a categorical distribution over returns (C51).
# 4. Noisy Networks: NoisyLinear layers instead of epsilon-greedy exploration.
# 5. Prioritized Replay Buffer: Sample more from high-error transitions.
############################################

# Noisy linear layer for Noisy DQN
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

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init * mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init * mu_range)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())


# Prioritized Replay Buffer
class SumTree:
    # SumTree for prioritized experience replay
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, p, data):
        idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, v):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            else:
                if v <= self.tree[left]:
                    idx = left
                else:
                    v -= self.tree[left]
                    idx = right

    @property
    def total_p(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    # Prioritized replay buffer with alpha and beta scheduling
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.max_p = 1.0

    def store(self, experience):
        p = self.max_p
        self.tree.add(p, experience)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total_p / batch_size
        beta = self._beta_by_frame(self.frame)
        self.frame += 1

        priorities = []
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx = self.tree.get_leaf(s)
            data_idx = idx - self.tree.capacity + 1
            exp = self.tree.data[data_idx]
            p = self.tree.tree[idx]
            batch.append(exp)
            idxs.append(idx)
            priorities.append(p)

        sampling_prob = priorities / self.tree.total_p
        is_weights = np.power(len(self) * sampling_prob, -beta)
        is_weights /= is_weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), idxs, is_weights

    def update_priorities(self, idxs, errors):
        epsilon = 1e-5
        for idx, error in zip(idxs, errors):
            p = (abs(error) + epsilon) ** self.alpha
            self.tree.update(idx, p)
            if p > self.max_p:
                self.max_p = p

    def __len__(self):
        if self.tree.data_pointer == 0 and np.all(self.tree.data == 0):
            return 0
        return min(self.tree.data_pointer, self.capacity) if self.tree.data_pointer != 0 else self.capacity

    def _beta_by_frame(self, frame):
        return min(1.0, self.beta_start + (1.0 - self.beta_start)*frame/self.beta_frames)


# Rainbow Network: Dueling + Noisy Layers + Distributional
class RainbowNetwork(nn.Module):
    # C51 distributional: use num_atoms, v_min, v_max to define the support
    def __init__(self, state_dim, action_dim, num_atoms=51, v_min=0.0, v_max=50.0):
        super(RainbowNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)

        # Feature layers
        self.fc1 = NoisyLinear(state_dim, 64)
        self.fc2 = NoisyLinear(64, 64)

        # Dueling streams (Noisy)
        self.value_stream = NoisyLinear(64, self.num_atoms)        # value distribution
        self.advantage_stream = NoisyLinear(64, action_dim * self.num_atoms)

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

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()

class RainbowAgent:
    # Key differences from DQN:
    # - Use RainbowNetwork (Noisy, Dueling, Distributional)
    # - Double DQN update
    # - Prioritized replay buffer
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-4,
                 num_atoms=51, v_min=0.0, v_max=50.0, n_step=4,
                 buffer_size=10000, batch_size=32, alpha=0.6, beta_start=0.4, beta_frames=100000,
                 epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0, device='cpu'):
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

        self.q_network = RainbowNetwork(state_dim, action_dim, num_atoms, v_min, v_max).to(self.device)
        self.target_q_network = RainbowNetwork(state_dim, action_dim, num_atoms, v_min, v_max).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

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

    # def store_experience(self, state, action, reward, next_state, done):
    #     self.replay_buffer.store((state, action, reward, next_state, done))

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in n-step buffer and replay buffer.
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step or done:
            n_step_state, n_step_action, n_step_return, n_step_next_state, n_step_done = self._get_n_step_info()
            self.replay_buffer.store((n_step_state, n_step_action, n_step_return, n_step_next_state, n_step_done))

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

    def project_distribution(self, next_dist, rewards, dones):
        # next_dist: [batch, num_atoms] (already chosen by next_actions)
        batch_size = rewards.size(0)
        projected_dist = torch.zeros((batch_size, self.num_atoms), device=next_dist.device)
        # Categorical projection of the Bellman update
        for i in range(self.num_atoms):
            Tz = rewards + (1 - dones)*self.gamma*self.support[i]
            Tz = torch.clamp(Tz, self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            eq_mask = (l == u)
            not_eq_mask = ~eq_mask

            # Distribute probability
            if eq_mask.any():
                eq_idx = eq_mask.nonzero(as_tuple=True)[0]
                projected_dist[eq_idx, l[eq_idx]] += next_dist[eq_idx, i]

            if not_eq_mask.any():
                neq_idx = not_eq_mask.nonzero(as_tuple=True)[0]
                l_neq = l[neq_idx]
                u_neq = u[neq_idx]
                projected_dist[neq_idx, l_neq] += next_dist[neq_idx, i]*(u_neq.float()-b[neq_idx])
                projected_dist[neq_idx, u_neq] += next_dist[neq_idx, i]*(b[neq_idx]-l_neq.float())

        return projected_dist
    
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

        states, actions, rewards, next_states, dones, idxs, is_weights = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        # Noisy reset
        self.q_network.reset_noise()
        self.target_q_network.reset_noise()

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
        loss = (loss_element * is_weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        errors = loss_element.detach().cpu().numpy()
        self.replay_buffer.update_priorities(idxs, errors)

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        # Not needed for NoisyNet, but keep for compatibility
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)