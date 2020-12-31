import numpy as np
import torch


class ReplayBuffer:

    def __init__(self, state_size, action_size, capacity, device, seed):
        self.seed = torch.manual_seed(seed)
        np.random.seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.capacity = capacity
        self.device = device

        self.index = 0
        self.full = False

        self.state = np.empty(shape=(capacity, *state_size), dtype=np.float32)
        self.action = np.empty(shape=(capacity, *action_size), dtype=np.float32)
        self.next_state = np.empty(shape=(capacity, *state_size), dtype=np.float32)
        self.reward = np.empty(shape=(capacity, 1), dtype=np.float32)
        self.done = np.empty(shape=(capacity, 1), dtype=np.int8)

    def add(self, state, action, next_state, reward, done):
        np.copyto(self.state[self.index], state)
        np.copyto(self.action[self.index], action)
        np.copyto(self.next_state[self.index], next_state)
        np.copyto(self.reward[self.index], reward)
        np.copyto(self.done[self.index], done)

        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def sample(self, batchsize):
        limit = self.index if not self.full else self.capacity
        batch = np.random.randint(0, limit, size=batchsize)

        state = torch.as_tensor(self.state[batch], device=self.device)
        action = torch.as_tensor(self.action[batch], device=self.device)
        next_state = torch.as_tensor(self.next_state[batch], device=self.device)
        reward = torch.as_tensor(self.reward[batch], device=self.device)
        done = torch.as_tensor(self.done[batch], device=self.device)

        return state, action, next_state, reward, done

    def save_memory(self):
        pass

    def load_memory(self):
        pass
