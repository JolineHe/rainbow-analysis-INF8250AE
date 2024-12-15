import torch
import torch.nn as nn
import torch.optim as optim

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

class A3CAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, rollout_steps=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.rollout_steps = rollout_steps

        self.q_network = ActorCriticNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_dones = []
        self.last_state = None
        self.last_action = None

        self.epsilon = 0.0
        self.epsilon_min = 0.0
        self.epsilon_decay = 1.0

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            policy_logits, _ = self.q_network(state_tensor)
            probs = torch.softmax(policy_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action

    def store_experience(self, state, action, reward, next_state, done):
        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_dones.append(done)

    def train(self, batch_size=32):
        if len(self.buffer_states) < self.rollout_steps and not (len(self.buffer_states) > 0 and self.buffer_dones[-1]):
            return

        states = torch.FloatTensor(self.buffer_states)
        actions = torch.LongTensor(self.buffer_actions).unsqueeze(1)
        rewards = self.buffer_rewards
        dones = self.buffer_dones

        with torch.no_grad():
            if dones[-1]:
                bootstrap_value = 0.0
            else:
                next_state = torch.FloatTensor(self.buffer_states[-1]).unsqueeze(0)
                _, next_value = self.q_network(next_state)
                bootstrap_value = next_value.item()

        returns = []
        R = bootstrap_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.append(R)
        returns.reverse()
        returns = torch.FloatTensor(returns).unsqueeze(1)

        policy_logits, values = self.q_network(states)
        log_probs = torch.log_softmax(policy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions)

        advantages = returns - values
        value_loss = advantages.pow(2).mean()
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_dones = []

    def update_target_network(self):
        pass

    def decay_epsilon(self):
        pass