import numpy as np
import gym
import torch
import torch.nn.functional as F
from collections import deque
from tensorboardX import SummaryWriter
import json
import datetime


from models import Actor, DoubleQNetworks
from replaybuffer import ReplayBuffer


class Agent():

    def __init__(self, action_size, state_size, config, seed):
        self.action_size = action_size
        self.state_size = state_size
        self.tau = config["TD3_tau"]
        self.gamma = config["TD3_gamma"]
        self.batch_size = config["TD3_batch_size"]
        self.noise_clip = config["TD3_noise_clip"]
        self.sigma_exploration = config["TD3_sigma_exploration"]
        self.sigma_tilde = config["TD3_sigma_tilde"]
        self.update_freq = config["TD3_update_freq"]
        self.max_action = config["max_action"]
        self.min_action = config["min_action"]
        self.seed = seed
        self.env = gym.make("LunarLanderContinuous-v2")

        # set seeds for comparison
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)

        # check whether cuda available if chosen as device
        if config["device"] == "cuda":
            if not torch.cuda.is_available():
                config["device"] == "cpu"
        self.device = config["device"]

        # replay
        self.memory = ReplayBuffer((state_size, ), (action_size, ),
                                   config["buffer_size"], self.device, config["seed"])

        # everything necessary for SummaryWriter
        time_for_path = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        pathname = f" {time_for_path}"
        tensorboard_name = str(config["locexp"]) + '/runs' + str(pathname)
        self.writer = SummaryWriter(tensorboard_name)
        self.steps = 0

        # actor, optimizer of actor, target for actor, critic, optimizer of critic, target for critic
        self.actor = Actor(state_size, action_size, config["fc1_units"],
                           config["fc2_units"], config["seed"]).to(self.device)
        self.optimizer_a = torch.optim.Adam(self.actor.parameters(),
                                            config["TD3_lr_actor"])
        self.target_actor = Actor(state_size, action_size, config["fc1_units"],
                                  config["fc2_units"], config["seed"]).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic = DoubleQNetworks(state_size, action_size, config["fc1_units"],
                                      config["fc2_units"], config["seed"]).to(self.device)
        self.optimizer_q = torch.optim.Adam(self.critic.parameters(),
                                            config["TD3_lr_critic"])
        self.target_critic = DoubleQNetworks(state_size, action_size, config["fc1_units"],
                                             config["fc2_units"], config["seed"]).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def act(self, state, greedy=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        state = state.unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).numpy()[0]
            # ^ torch.argmax(q_nns) in continuous case
        return action

    def train(self, episodes, timesteps):
        mean_r = 0
        mean_episode = 0
        dq = deque(maxlen=100)
        for i in range(episodes):
            state = self.env.reset()

            for t in range(1, timesteps+1):
                noise = np.zeros(shape=(self.action_size,))
                for idx in range(len(noise)):
                    noise[idx] = np.random.normal(0, self.sigma_exploration * self.max_action)
                action = np.clip((self.act(state) + noise), self.min_action, self.max_action)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.add(state, action, next_state, reward, done)
                state = next_state
                mean_r += reward

                # fill replay buffer with 10 samples before updating the policy
                if i > 10:
                    self.update(t)

                if done:
                    print(f"timesteps until break: {t}")
                    break

            # print and write data to tensorboard for pre_evaluation
            dq.append(mean_r)
            mean_episode = np.mean(dq)
            self.writer.add_scalar("average_reward", mean_episode, self.steps)
            print(f"Episode: {i}, mean_r: {mean_r}, mean_episode: {mean_episode}")

            mean_r = 0

    def update(self, t):
        self.steps += 1
        # sample minibatch and calculate target value and q_nns
        state, action, next_state, reward, done = self.memory.sample(self.batch_size)

        noise = np.clip((torch.randn_like(action, dtype=torch.float32) * self.sigma_tilde), -self.noise_clip, self.noise_clip)
        #noise = (torch.randn_like(action, dtype=torch.float32) * self.sigma_tilde).clamp(-self.noise_clip, self.noise_clip)
        with torch.no_grad():
            next_action = (self.target_actor(next_state) + noise).clamp(self.min_action, self.max_action)
            q_target = self.target_critic(next_state, next_action)
            y_target = reward + (self.gamma * torch.min(q_target[0], q_target[1]) * (1-done))

        # update critic
        q_samples_target = self.critic(state, action)
        loss_0 = F.mse_loss(y_target, q_samples_target[0])
        loss_1 = F.mse_loss(y_target, q_samples_target[1])
        critics_loss = loss_0 + loss_1
        self.writer.add_scalar("critics_loss", critics_loss, self.steps)

        # set gradients to zero and optimize q
        self.optimizer_q.zero_grad()
        critics_loss.backward()
        self.optimizer_q.step()

        if t % self.update_freq == 0:

            # update actor
            c_action = self.actor(state)
            q_sum_samples = self.critic(state, c_action)[0]
            actor_loss = -q_sum_samples.mean()
            self.writer.add_scalar("actor_loss", actor_loss, self.steps)

            # set gradients to zero and optimize a
            self.optimizer_a.zero_grad()
            actor_loss.backward()
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

    env = gym.make(config["env_name"])

    action_space = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]
    config["max_action"] = env.action_space.high[0]
    config["min_action"] = env.action_space.low[0]

    agent = Agent(action_size=action_space, state_size=state_size,
                  config=config, seed=config["seed"])

    agent.train(episodes=1000, timesteps=1000)


if __name__ == "__main__":
    main()
