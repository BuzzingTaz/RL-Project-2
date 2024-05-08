import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.buffer_size = max_size
        self.count = 0
        self.states = np.zeros((self.buffer_size, input_shape[0]), dtype=np.float32) 
        self.actions = np.zeros((self.buffer_size, input_shape[1]), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.new_states = np.zeros((self.buffer_size, input_shape[0]), dtype=np.float32)
        self.terminal = np.zeros(self.buffer_size)

    def add_sample(self, state, action, reward, new_state, done):
        index = self.count % self.buffer_size
        self.states[index] = state
        self.new_states[index] = new_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.terminal[index] = 1 - done
        self.count += 1

    def return_sample(self, batch_size):
        max_choice = min(self.count, self.buffer_size)

        batch = np.random.choice(max_choice, batch_size)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        new_states = self.new_states[batch]
        terminal = self.terminal[batch]

        return states, actions, rewards, new_states, terminal