import torch
import random
from agents.dqn import DQNAgent
import numpy as np

class DoubleDQNAgent(DQNAgent):
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

        # gather(1,actions) collects Q-values from q_values based on indices in actions
        # For example, if actions=[1,0,2], it will collect Q-values for actions 1,0,2 from each state's Q-value vector
        # 1 means gathering along dimension 1 (action dimension), dimension 0 is the batch dimension
        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.q_network(next_states)
        max_actions = next_q_values.max(1)[1].unsqueeze(1)
        next_q_values = self.target_q_network(next_states).gather(1, max_actions)
        
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    