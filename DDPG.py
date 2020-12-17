import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque


class Agent():
    


class Actor(nn.Module):

    def __init__(self, state_size, action_size, fc1, fc2, fc3):
        super(nn.Module, self).__init__()
        # fci = fully connected i
        self.layer1 = nn.Linear(state_size, action_size)
        self.layer2 = nn.Linear(fc1, fc2)
        self.layer3 = nn.Linear(fc2, fc3)

    def forward(self, states):
        out_l1 = nn.relu(self.layer1(states))
        out_l2 = nn.relu(self.layer2(out_l1))
        out = nn.Tanh(self.layer3(out_l2))
        return out


class Q_network(nn.Module):

    def __init__(self, state_size, action_size, fc1, fc2, fc3):
        super(nn.Module, self).__init__()
        # fci = fully connected i
        self.layer1 = nn.Linear(state_size + action_size, fc1)
        self.layer2 = nn.Linear(fc1, fc2)
        self.layer3 = nn.Linear(fc2, fc3)

    def forward(self, states, actions):
        batch = torch.cat((states, actions), 1)
        out_l1 = nn.relu(self.layer1(batch))
        out_l2 = nn.relu(self.layer2(out_l1))
        out = self.layer3(out_l2)
        return out


class ReplayBuffer:

    def __init__(self, state_shape, action_size, capacity, device):
        self.state_size = state_size
        self.action_size = action_size
        self.capacity = capacity
        self.device = device

        self.index = 0
        self.full = False

        self.state = np.empty(shape=(capacity, *state_shape), dtype=np.int8)
        self.action = np.empty(shape=(capacity, *action_size), dtype=np.int8)
        self.next_state = np.empty(shape=(capacity, *state_shape), dtype=np.int8)
        self.reward = np.empty(shape=(capacity, 1), dtype=np.int8)
        self.done = np.empty(shape=(capacity, 1), dtype=np.int8)

    def add(self, state, reward, action, next_state):
        self.state[self.index] = state
        self.action[self.index] = action
        self.next_state[self.index] = next_state
        self.reward[self.index] = reward

        self.index = (self.index + 1) % self.capacity
        self.full = True if self.index == 0 else False

    def sample(self, batchsize):
        limit = self.index if not self.full else self.capacity - 1

        batch = np.random.randint(0, limit, size=batchsize)

        state = torch.as_tensor(self.state[batch])
        action = torch.as_tensor(self.action[batch])
        next_state = torch.as_tensor(self.next_state[batch])
        reward = torch.as_tensor(self.reward[batch])
        done = torch.as_tensor(self.done[batch])

        return state, action, next_state, reward, done

    def save_memory(self):
        pass

    def load_memory(self):
        pass
