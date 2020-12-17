#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from tensorboardX import SummaryWriter

import json

class OrnsteinUhlenbeckProcess:
    def __init__(self, mu=np.zeros(1), sigma=0.05, theta=.25, dimension=1e-2, x0=None, num_steps=12000):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dimension
        self.x0 = x0
        self.reset()

    def step(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


# In[113]:


class Actor(nn.Module):

    def __init__(self, state_size, action_size, fc1, fc2):
        super(Actor, self).__init__()
        # fci = fully connected i
        self.layer1 = nn.Linear(state_size, fc1)
        self.layer2 = nn.Linear(fc1, fc2)
        self.layer3 = nn.Linear(fc2, action_size)

        self.leak = 0.01

        self.reset_parameteres()

    def forward(self, states):
        out_l1 = F.relu(self.layer1(states))
        out_l2 = F.relu(self.layer2(out_l1))
        out = F.tanh(self.layer3(out_l2))
        #out = F.tanh(self.layer3(out_l2))
        return out

    def reset_parameteres(self):
        torch.nn.init.kaiming_normal_(self.layer1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layer2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.layer3.weight.data, -3e-3, 3e-3)


class Q_network(nn.Module):

    def __init__(self, state_size, action_size, fc1, fc2):
        super(Q_network, self).__init__()
        # fci = fully connected i
        self.layer1 = nn.Linear(state_size + action_size, fc1)
        self.layer2 = nn.Linear(fc1, fc2)
        self.layer3 = nn.Linear(fc2, 1)

        self.leak = 0.01

        self.reset_parameteres()

    def forward(self, states, actions):
        batch = torch.cat([states, actions], 1)
        out_l1 = F.relu(self.layer1(batch))
        out_l2 = F.relu(self.layer2(out_l1))
        out = self.layer3(out_l2)
        return out

    def reset_parameteres(self):
        torch.nn.init.kaiming_normal_(self.layer1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layer2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.layer3.weight.data, -3e-3, 3e-3)

class ReplayBuffer:

    def __init__(self, state_size, action_size, capacity, device):
        self.state_size = state_size
        self.action_size = action_size
        self.capacity = capacity
        self.device = device

        self.index = 0
        self.full = False

        self.state = np.empty(shape=(capacity, state_size), dtype=np.float32)
        self.action = np.empty(shape=(capacity, action_size), dtype=np.float32)
        self.next_state = np.empty(shape=(capacity, state_size), dtype=np.float32)
        self.reward = np.empty(shape=(capacity, 1), dtype=np.float32)
        self.done = np.empty(shape=(capacity, 1), dtype=np.int8)

    def add(self, state, action, next_state, reward, done):
        np.copyto(self.state[self.index], state)
        np.copyto(self.action[self.index], action)
        np.copyto(self.next_state[self.index], next_state)
        np.copyto(self.reward[self.index], reward)
        np.copyto(self.done[self.index], done)

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


# In[114]:


class Agent():

    def __init__(self, action_size, state_size, config):
        self.action_size = action_size
        self.state_size = state_size
        self.seed = config["seed"]
        self.lr_actor = config["lr_actor"]
        self.lr_critic = config["lr_critic"]
        self.tau = config["tau"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        if config["device"] == "cuda":
            if torch.cuda.is_available():
                config["device"] == "cpu"
        self.device = config["device"]

        # initialize noise
        self.noise = OrnsteinUhlenbeckProcess(sigma=0.2, theta=0.15)
        self.noise.reset()

        # replay
        self.memory = ReplayBuffer(state_size, action_size, config["buffer_size"], self.device)

        # everything necessary for SummaryWriter
        pathname = 8
        tensorboard_name = str(config["locexp"]) + '/runs' + str(pathname)
        self.writer = SummaryWriter(tensorboard_name)
        self.steps = 0

        # set seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # actor, optimizer of actor, target for actor, critic, optimizer of critic, target for critic
        self.actor = Actor(state_size, action_size, config["fc1_units"], config["fc2_units"]).to(self.device)
        self.optimizer_a = torch.optim.Adam(self.actor.parameters(), self.lr_actor)
        self.target_actor = Actor(state_size, action_size, config["fc1_units"], config["fc2_units"]).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic = Q_network(state_size, action_size, config["fc1_units"], config["fc2_units"]).to(self.device)
        self.optimizer_q = torch.optim.Adam(self.critic.parameters(), self.lr_critic)
        self.target_critic = Q_network(state_size, action_size, config["fc1_units"], config["fc2_units"]).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def act(self, state, greedy=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=torch.device("cpu"))
        state = state.unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).numpy()[0]#.detach().numpy()

        # ^ torch.argmax(q_nns) in continuous case
        noise = self.noise.step()
        #print(noise)
        action = action if greedy else np.clip(action + noise, -1, 1)#self.noise.step(), -1, 1)
        return action

    def train(self, episodes, timesteps):
        #env = gym.make("MountainCarContinuous-v0")
        env = gym.make("LunarLanderContinuous-v2")

        mean_r = 0
        mean_episode = 0
        dq = deque(maxlen=100)
        for i in range(1, episodes+1):
            state = env.reset()
            if i % 10 == 0:
                self.noise.reset()

            for j in range(timesteps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                #next_state = np.squeeze(next_state, axis=1) bei Car
                self.memory.add(state, action, next_state, reward, done)
                state = next_state
                mean_r += reward

                # fill replay buffer with 10 samples before updating the policy
                if i > 10:
                    self.update()

                if done:
                    print(f"timesteps until break: {j}")
                    break

            # append mean of episode to list and add average of last 10 means if existent, then reset mean
            dq.append(mean_r)
            mean_episode = np.mean(dq)
            self.writer.add_scalar("a_rew", mean_episode, i)

            # print every tenth episode to keep track
            print(f"Episode: {i}, mean_r: {mean_r}, mean_episode: {mean_episode}")
            mean_r = 0

    def update(self):
        self.steps += 1
        # sample minibatch and calculate target value and q_nns
        state, action, next_state, reward, done = self.memory.sample(self.batch_size)

        with torch.no_grad():
            next_action = self.target_actor(next_state)#.detach()#.numpy()
            q_target = self.target_critic(next_state, next_action)
            y_target = reward + (self.gamma * q_target * (1-done))

        # update critic
        q_samples_target = self.critic(state, action)
        loss_critic = F.mse_loss(y_target, q_samples_target)
        self.writer.add_scalar("loss_critic", loss_critic, self.steps)

        # set gradients to zero and optimize q
        self.optimizer_q.zero_grad()
        loss_critic.backward()
        self.optimizer_q.step

        # update actor
        c_action = self.actor(state)
        q_sum_samples = self.critic(state, c_action)
        loss_actor = -q_sum_samples.mean()
        self.writer.add_scalar("loss_actor", loss_actor, self.steps)

        # set gradients to zero and optimize a
        self.optimizer_a.zero_grad()
        loss_actor.backward()
        self.optimizer_a.step()

        # update target networks
        self.update_target(self.actor, self.target_actor)
        self.update_target(self.critic, self.target_critic)

    def update_target(self, online, target):
        for parameter, target in zip(online.parameters(), target.parameters()):
            target.data.copy_(self.tau * parameter.data + (1 - self.tau) * target.data)


# In[11]:




def main():
    with open('param.json') as f:
        config = json.load(f)

        #env = gym.make("MountainCarContinuous-v0")
    env = gym.make("LunarLanderContinuous-v2")
    state = env.reset()
    action_space = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    agent = Agent(action_size=action_space, state_size=state_size, config=config)

    agent.train(episodes=1000, timesteps=1000)

if __name__ == "__main__":
    main()
