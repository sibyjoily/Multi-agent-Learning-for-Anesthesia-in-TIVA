import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal
import numpy as np
from collections import namedtuple, deque
import random

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        # experiences = self.memory
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


class DuelingNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = torch.relu(self.fc3(x))
        advantage = torch.relu(self.fc4(x))
        q_values = value + (advantage - advantage.mean(1, keepdim=True))
        return q_values








class DQNNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x





class Actor_maddpg(nn.Module):
    def __init__(self, state_size, action_size,seed):
        super(Actor_maddpg, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.reset_parameters()
        # self.max_value = max_value

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        # Scale and clamp the output to [0, max_value]
        # x = (x + 1) / 2 * self.max_value
        # x = torch.clamp(x, min=torch.zeros_like(x), max=self.max_value)
        return x

class Critic_maddpg(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Critic_maddpg, self).__init__()
        self.fcs1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256 + action_size, 128)
        self.fc3 = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = torch.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Actor_coma(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Actor_coma, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.reset_parameters()
        # self.max_value = max_value

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        # Scale and clamp the output to [0, max_value]
        # x = (x + 1) / 2 * self.max_value
        # x = torch.clamp(x,  min=torch.zeros_like(x), max=self.max_value)
        return x

class Critic_coma(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Critic_coma, self).__init__()
        self.fcs1 = nn.Linear(state_size + action_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        xs = torch.cat((states, actions), dim=1)
        xs = torch.relu(self.fcs1(xs))
        x = torch.relu(self.fc2(xs))
        return self.fc3(x)

# 定义 VDN 网络 （（使用离散动作空间Discrete Action Spaces））
class VDNNet(nn.Module):
    def __init__(self, state_dim, action_dim1,action_dim2):
        super(VDNNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim1)
        self.fc4 = nn.Linear(128, action_dim2)
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    #     self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    #     self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    #     self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_value1 = torch.sigmoid(self.fc3(x)) # 第一个智能体的动作值
        action_value2 = torch.sigmoid(self.fc4(x))  # 第二个智能体的动作值
        return action_value1, action_value2

# 定义 QMIX 网络 （（使用离散动作空间Discrete Action Spaces））
class qmixNet(nn.Module):
    def __init__(self, state_dim, action_dim1,action_dim2):
        super(qmixNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim1)
        self.fc4 = nn.Linear(128, action_dim2)
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    #     self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    #     self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    #     self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_value1 = torch.tanh(self.fc3(x)) # 第一个智能体的动作值
        action_value2 = torch.tanh(self.fc4(x))  # 第二个智能体的动作值
        return action_value1, action_value2



class QMIX(nn.Module):
    def __init__(self, n_agents, state_dim):
        super(QMIX, self).__init__()
        self.hyper_w1 = nn.Linear(state_dim, 2)
        self.hyper_b1 = nn.Linear(state_dim, 1)

    def forward(self, agent_qs, states):
        batch_size = agent_qs.size(0)
        n_agents = agent_qs.size(1)

        # First layer weights and biases
        w1 = torch.abs(self.hyper_w1(states)).view(-1, 2,1)
        b1 = self.hyper_b1(states).view(-1, 1,1)

        # Ensure agent_qs is 3D
        agent_qs = agent_qs.view(batch_size, 1,n_agents)

        # Mixing network with one layer
        hidden = torch.bmm(agent_qs, w1) + b1

        # # Optionally apply an activation function
        # q_tot = F.elu(hidden)

        return hidden.view(batch_size, -1)





def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
