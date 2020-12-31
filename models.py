import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, state_size, action_size, fc1, fc2, seed):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer1 = nn.Linear(state_size, fc1)
        self.layer2 = nn.Linear(fc1, fc2)
        self.layer3 = nn.Linear(fc2, action_size)

        self.leak = 0.01
        self.reset_parameteres()

    def forward(self, states):
        out_l1 = F.relu(self.layer1(states))
        out_l2 = F.relu(self.layer2(out_l1))
        out = F.tanh(self.layer3(out_l2))
        return out

    def reset_parameteres(self):
        nn.init.kaiming_normal_(self.layer1.weight.data, a=self.leak, mode='fan_in')
        nn.init.kaiming_normal_(self.layer2.weight.data, a=self.leak, mode='fan_in')
        nn.init.uniform_(self.layer3.weight.data, -3e-3, 3e-3)


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, fc1, fc2, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer1 = nn.Linear(state_size + action_size, fc1)
        self.layer2 = nn.Linear(fc1, fc2)
        self.layer3 = nn.Linear(fc2, 1)

        self.leak = 0.01
        self.reset_parameteres()

    def forward(self, states, actions):
        batch = torch.cat([states, actions], 1)
        x1 = F.relu(self.layer1(batch))
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2)
        return x3

    def reset_parameteres(self):
        nn.init.kaiming_normal_(self.layer1.weight.data, a=self.leak, mode='fan_in')
        nn.init.kaiming_normal_(self.layer2.weight.data, a=self.leak, mode='fan_in')
        nn.init.uniform_(self.layer3.weight.data, -3e-3, 3e-3)


class DoubleQNetworks(nn.Module):

    def __init__(self, state_size, action_size, fc1, fc2, seed):
        super(DoubleQNetworks, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer1_1 = nn.Linear(state_size + action_size, fc1)
        self.layer2_1 = nn.Linear(fc1, fc2)
        self.layer3_1 = nn.Linear(fc2, 1)

        self.layer1_2 = nn.Linear(state_size + action_size, fc1)
        self.layer2_2 = nn.Linear(fc1, fc2)
        self.layer3_2 = nn.Linear(fc2, 1)

        self.leak = 0.01

        self.reset_parameteres()

    def forward(self, states, actions):
        batch = torch.cat([states, actions], 1)
        q1_out1 = F.relu(self.layer1_1(batch))
        q1_out2 = F.relu(self.layer2_1(q1_out1))
        q1_out3 = self.layer3_1(q1_out2)

        q2_out1 = F.relu(self.layer1_2(batch))
        q2_out2 = F.relu(self.layer2_2(q2_out1))
        q2_out3 = self.layer3_2(q2_out2)

        return q1_out3, q2_out3

    def reset_parameteres(self):
        torch.nn.init.kaiming_normal_(self.layer1_1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layer2_1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.layer3_1.weight.data, -3e-3, 3e-3)

        torch.nn.init.kaiming_normal_(self.layer1_2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.layer2_2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.layer3_2.weight.data, -3e-3, 3e-3)
