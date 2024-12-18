import torch
import numpy as np
import random
from agents.ddqn import DoubleDQNAgent


class SumTree:
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
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.capacity = capacity
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
            experience = self.tree.data[data_idx]
            p = self.tree.tree[idx]
            batch.append(experience)
            idxs.append(idx)
            priorities.append(p)

        sampling_probabilities = priorities / self.tree.total_p
        is_weight = np.power(len(self) * sampling_probabilities, -beta)
        is_weight /= is_weight.max()
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(
            dones), idxs, is_weight

    def update_priorities(self, idxs, errors):
        epsilon = 1e-5
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + epsilon) ** self.alpha
            self.tree.update(idx, p)
            if p > self.max_p:
                self.max_p = p

    def __len__(self):
        # 当数据还未填满时，长度为data_pointer，否则为capacity
        return min(self.tree.data_pointer, self.capacity) if self.tree.data_pointer != 0 else self.capacity

    def _beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * frame_idx / self.beta_frames)

class PrioritizedDoubleDQNAgent(DoubleDQNAgent):
    def __init__(self, state_dim, action_dim, gamma=0.9, lr=0.001,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=2000, alpha=0.6, beta_start=0.4, beta_frames=100000, device='cpu'):
        super().__init__(state_dim, action_dim, gamma, lr, epsilon, epsilon_min, epsilon_decay, device)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha, beta_start, beta_frames)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.store((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones, idxs, is_weights = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)

        next_q_values_online = self.q_network(next_states)
        max_actions = next_q_values_online.max(1)[1].unsqueeze(1)
        next_q_values_target = self.target_q_network(next_states).gather(1, max_actions)

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values_target.detach()

        errors = (q_values.detach() - target_q_values).cpu().numpy()
        loss_element_wise = self.loss_fn(q_values, target_q_values)
        loss = (loss_element_wise * is_weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(idxs, errors)