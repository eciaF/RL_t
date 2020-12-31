import time
import numpy as np
import gym
import torch
import torch.nn.functional as F
from collections import deque
from tensorboardX import SummaryWriter
import json

from helper import OrnsteinUhlenbeckProcess
from models import Actor, QNetwork
from replaybuffer import ReplayBuffer


class Agent():

    def __init__(self, action_size, state_size, config):
        self.action_size = action_size
        self.state_size = state_size
        self.tau = config["tau"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]

        # check whether cuda available if chosen as device
        if config["device"] == "cuda":
            if not torch.cuda.is_available():
                config["device"] == "cpu"
        self.device = config["device"]

        # initialize noise
        self.noise = OrnsteinUhlenbeckProcess(sigma=0.2, theta=0.15,
                                              dimension=action_size)
        self.noise.reset()

        # replay
        self.memory = ReplayBuffer((state_size, ), (action_size, ),
                                   config["buffer_size"], self.device)

        # everything necessary for SummaryWriter
        pathname = f"tau={self.tau}, gamma: {self.gamma}, \
                   batchsize: {self.batch_size}, {time.ctime()}"
        tensorboard_name = str(config["locexp"]) + '/runs' + str(pathname)
        self.writer = SummaryWriter(tensorboard_name)
        self.steps = 0

        # actor, optimizer of actor, target for actor, critic, optimizer of
        #  critic, target for critic
        self.actor = Actor(state_size, action_size, config["fc1_units"],
                           config["fc2_units"]).to(self.device)
        self.optimizer_a = torch.optim.Adam(self.actor.parameters(),
                                            config["lr_actor"])
        self.target_actor = Actor(state_size, action_size, config["fc1_units"],
                                  config["fc2_units"]).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic = QNetwork(state_size, action_size, config["fc1_units"],
                               config["fc2_units"]).to(self.device)
        self.optimizer_q = torch.optim.Adam(self.critic.parameters(),
                                            config["lr_critic"])
        self.target_critic = QNetwork(state_size, action_size,
                                      config["fc1_units"],
                                      config["fc2_units"]).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def act(self, state, greedy=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        state = state.unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).numpy()[0]
            # ^ torch.argmax(q_nns) in continuous case
        noise = self.noise.step()
        action = action if greedy else np.clip(action + noise, -1, 1)
        return action

    def train(self, episodes, timesteps):
        env = gym.make("LunarLanderContinuous-v2")

        mean_r = 0
        mean_episode = 0
        dq = deque(maxlen=100)
        for i in range(episodes):
            state = env.reset()
            if i % 10 == 0:
                self.noise.reset()

            for j in range(timesteps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.memory.add(state, action, next_state, reward, done)
                state = next_state
                mean_r += reward

                # fill replay buffer with 10 samples before updating the policy
                if i > 10:
                    self.update()

                if done:
                    print(f"timesteps until break: {j}")
                    break

            # print and write data to tensorboard for pre_evaluation
            dq.append(mean_r)
            mean_episode = np.mean(dq)
            self.writer.add_scalar("a_rew", mean_episode, i)
            print(f"Episode: {i}, mean_r: {mean_r}, \
                    mean_episode: {mean_episode}")

            mean_r = 0

    def update(self):
        self.steps += 1
        # sample minibatch and calculate target value and q_nns
        state, action, next_state, reward, done = self.memory.sample(self.batch_size)

        with torch.no_grad():
            next_action = self.target_actor(next_state)
            q_target = self.target_critic(next_state, next_action).detach()
            y_target = reward + (self.gamma * q_target * (1-done))

        # update critic
        q_samples_target = self.critic(state, action)
        loss_critic = F.mse_loss(y_target, q_samples_target)
        self.writer.add_scalar("loss_critic", loss_critic, self.steps)

        # set gradients to zero and optimize q
        self.optimizer_q.zero_grad()
        loss_critic.backward()
        self.optimizer_q.step()

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
            target.data.copy_(self.tau * parameter.data +
                              (1 - self.tau) * target.data)


def main():
    with open('param.json') as f:
        config = json.load(f)

    env = gym.make("LunarLanderContinuous-v2")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    env.seed(config["seed"])
    env.action_space.seed(config["seed"])

    env.reset()
    action_space = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    agent = Agent(action_size=action_space, state_size=state_size,
                  config=config)

    agent.train(episodes=1000, timesteps=1000)


if __name__ == "__main__":
    main()
