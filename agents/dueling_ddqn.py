import torch
import random
from agents.dqn import DQNAgent

class DuelingDDQNAgent(DQNAgent):
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

        # Dueling DQN architecture splits Q-values into value and advantage streams
        # Value stream estimates state value V(s)
        # Advantage stream estimates advantages A(s,a) for each action
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        
        # Get Q-values from main network
        q_values = self.q_network(states)
        value = q_values.mean(dim=1, keepdim=True)  # V(s)
        advantage = q_values - value  # A(s,a)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        q_values = q_values.gather(1, actions)

        # Get next Q-values from target network
        next_q_values = self.q_network(next_states)
        next_value = next_q_values.mean(dim=1, keepdim=True)
        next_advantage = next_q_values - next_value
        next_q_values = next_value + (next_advantage - next_advantage.mean(dim=1, keepdim=True))
        
        # Double DQN: Use main network to select actions, target network to evaluate them
        max_actions = next_q_values.max(1)[1].unsqueeze(1)
        next_q_values = self.target_q_network(next_states)
        next_value = next_q_values.mean(dim=1, keepdim=True)
        next_advantage = next_q_values - next_value
        next_q_values = next_value + (next_advantage - next_advantage.mean(dim=1, keepdim=True))
        next_q_values = next_q_values.gather(1, max_actions)
        
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()