import gym


class CartPoleEnv:
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode='rgb_array')
        self.action_space = [0, 1]
        self.num_states = 4
        self.state = self.env.reset()
        self.done = False

    def set_state(self, state):
        self.state = state
        self.env.state = state
        return self.state, {}

    def reset(self):
        self.state = self.env.reset()
        self.done = False
        return self.state, {}

    def step(self, action):
        assert action in self.action_space, "Invalid action"
        self.env.step(action)
        state, reward, done, _, info = self.env.step(action)
        self.state = state
        self.done = done
        return self.state, reward, self.done, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()