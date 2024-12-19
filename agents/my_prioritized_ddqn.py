import torch
import numpy as np
import random
from agents.ddqn import DoubleDQNAgent

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data_pointer = 0
        self.n_entries = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self):
        return int(self.tree[0])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=1000):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.abs_err_upper = 1.
        self.epsilon = 0.01
        self.beta_increment_per_sampling = 1/beta_frames
        self.alpha = alpha
        self.beta = beta_start
        self.abs_err_upper = 1.

    def __len__(self):
        return self.tree.total()

    def push(self, error, sample):
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(p, sample)         

    def sample(self, batch_size):
        pri_segment = self.tree.total() / batch_size
        priorities = []
        batch = []
        idxs = []
        is_weights = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = pri_segment * i
            b = pri_segment * (i+1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        return zip(*batch), idxs, is_weights
    
    def batch_update(self, tree_idx, abs_errors):#Update the importance sampling weight
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class PrioritizedDoubleDQNAgent(DoubleDQNAgent):
    def __init__(self, state_dim, action_dim, gamma=0.9, lr=0.001,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=2000, alpha=0.6, beta_start=0.4, beta_frames=1000, device='cpu'):
        super().__init__(state_dim, action_dim, gamma, lr, epsilon, epsilon_min, epsilon_decay, device)

        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha, beta_start, beta_frames)
    
    def store_experience(self, state, action, reward, next_state, done):
        # TD-error priority
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)[action]
            target_q_values = self.target_q_network(next_state_tensor)
            print(target_q_values.shape)

        transition = (state, action, reward, next_state, done)

        if done:
            error = abs(q_values - reward)
        else:
            error = abs(q_values - (reward + self.gamma * target_q_values.max().item()))

        error = error.detach().cpu().numpy()
        self.replay_buffer.push(error, transition)

    def train(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        batch, tree_idx, is_weights = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = map(np.array, batch)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(-1).to(self.device)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.q_network(next_states)
        max_actions = next_q_values.max(1)[1].unsqueeze(1)
        next_q_values = self.target_q_network(next_states).gather(1, max_actions)
        
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        errors = torch.abs(q_values.detach() - target_q_values).squeeze(-1).detach().cpu().numpy()
        loss = self.loss_fn(is_weights * q_values, is_weights * target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.batch_update(tree_idx, errors)


