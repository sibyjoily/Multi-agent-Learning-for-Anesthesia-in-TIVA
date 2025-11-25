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




# 定义 CW_QMIX 网络 （（使用离散动作空间Discrete Action Spaces））
class CW_qmixNet(nn.Module):
    def __init__(self, state_dim, action_dim1,action_dim2):
        super(CW_qmixNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim1)
        self.fc4 = nn.Linear(128, action_dim2)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_values1 = self.fc3(x)  # 第一个动作空间的动作值
        action_values2 = self.fc4(x)  # 第二个动作空间的动作值
        return action_values1, action_values2



def compute_cw_weights(q_values, q_tot, alpha):
    underestimation_mask = (q_values < q_tot).float()
    weights = alpha + (1.0 - alpha) * underestimation_mask
    return weights

class CW_QMIXNet(nn.Module):
    def __init__(self, agent_qs,state, alpha):
        super(CW_QMIXNet, self).__init__()
        self.alpha = alpha
        self.fc1 = nn.Linear(state, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, agent_qs,state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        weights = self.relu(self.fc3(x))
        q_tot = torch.sum(weights * agent_qs, dim=2, keepdim=True)
        ow_weights = compute_cw_weights(agent_qs, q_tot.detach(), self.alpha)
        q_tot = torch.sum(ow_weights * weights * agent_qs, dim=2, keepdim=True)
        return q_tot

# def compute_cw_weights(agent_qs, alpha):
#     max_q_value_indices = torch.argmax(agent_qs, dim=2)
#     weights = torch.full_like(agent_qs, alpha)
#     for i in range(agent_qs.shape[0]):
#         weights[i, :, max_q_value_indices[i]] = 1.0
#     return weights
#
# class CW_QMIXNet(nn.Module):
#     def __init__(self, agent_qs,state, alpha):
#         super(CW_QMIXNet, self).__init__()
#         self.alpha = alpha
#         self.agent_qs = agent_qs
#         self.fc1 = nn.Linear(state, 128)
#         # self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 2)
#
#     def forward(self, agent_qs,state):
#         cw_weights = compute_cw_weights(agent_qs, self.alpha)
#         agent_qs_weight = torch.relu(self.fc1(state))
#         # agent_qs_weight = torch.relu(self.fc2(agent_qs_weight))
#         agent_qs_weight = torch.relu(self.fc3(agent_qs_weight))
#
#         q_tot = torch.sum( cw_weights * agent_qs_weight*agent_qs, dim=2, keepdim=True)
#         # q_tot = torch.sum(cw_weights*agent_qs+agent_qs, dim=2, keepdim=True)
#         return q_tot





# 定义 QPLEX 网络 （（使用离散动作空间Discrete Action Spaces））
class qplexNet(nn.Module):
    def __init__(self, state_dim,action_dim1,action_dim2):
        super(qplexNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        # self.gru = nn.GRU(256, 128)
        self.fc3 = nn.Linear(128, action_dim1)
        self.fc4 = nn.Linear(128, action_dim2)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # x= self.gru(x)
        # action_values1 = self.fc2(x[0])  # 第一个动作空间的动作值
        # action_values2 = self.fc3(x[0])  # 第二个动作空间的动作值
        action_values1 = self.fc3(x)  # 第一个动作空间的动作值
        action_values2 = self.fc4(x)  # 第二个动作空间的动作值
        return action_values1, action_values2



class DuelingMixingNetwork(nn.Module):
    def __init__(self, agent_qs,state_dim):
        super(DuelingMixingNetwork, self).__init__()
        self.fc1 = nn.Linear(agent_qs, 128)
        self.fc2 = nn.Linear(state_dim, 128)
        self.fc3 = nn.Linear(128, 1)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, agent_qs,state):
        value = torch.relu(self.fc1(agent_qs))
        advantage = torch.tanh(self.fc2(state))
        value = torch.relu(self.fc3(value))
        advantage = torch.tanh(self.fc4(advantage))
        q_tot = value + advantage
        # value + (advantage - advantage.mean(1, keepdim=True))
        return q_tot


class TransformationNetwork(nn.Module):
    def __init__(self, state_dim):
        super(TransformationNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, 2)

    def forward(self, state, local_dueling):
        w = torch.relu(self.fc(state))
        w = w.unsqueeze(1).repeat(1, 2, 1)
        transformed_dueling =  local_dueling*w
        return transformed_dueling
#
# class DuelingMixingNetwork(nn.Module):
#     def __init__(self, input_dim, action_dim):
#         super(DuelingMixingNetwork, self).__init__()
#         self.fc_v = nn.Linear(input_dim, action_dim)
#         self.fc_a = nn.Linear(input_dim, action_dim)
#
#     def forward(self, transformed_dueling):
#         # Reshape the input to be two-dimensional
#         batch_size, _, _ = transformed_dueling.size()
#         transformed_dueling = transformed_dueling.view(batch_size, -1)
#         v = self.fc_v(transformed_dueling)
#         a = self.fc_a(transformed_dueling)
#         q_tot = v + (a-a.mean(1, keepdim=True))
#         return q_tot







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

class Actor_mappo(nn.Module):
    def __init__(self, state_size, action_size, seed,max_value):
        super(Actor_mappo, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))

        self.reset_parameters()
        self.max_value = max_value

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))  # 使用 sigmoid 将输出限制在 [-1, 1] 范围内
        mean = (mean + 1) / 2 * self.max_value
        mean = torch.clamp(mean, min=torch.zeros_like(mean),  max=self.max_value)
        return mean

    # def sample(self, state):
    #     mean = self.forward(state)
    #     std = torch.exp(self.log_std)
    #     normal = torch.distributions.Normal(mean, std)
    #     sample = normal.rsample()
    #
    #     # 使用 sigmoid 将采样的动作限制在 [-1, 1] 范围内
    #     action = torch.tanh(sample)
    #     # # 计算采样动作的对数概率，并进行修正以处理tanh变换的影响。
    #     log_prob = normal.log_prob(sample) - torch.log(1 - action.pow(2) + 1e-6)
    #     # # 计算采样动作的对数概率，并进行修正以处理sigmoid变换的影响。
    #     # log_prob = normal.log_prob(sample) - torch.log(action * (1 - action) + 1e-6)
    #
    #     return action, log_prob.sum(-1, keepdim=True)
#
class Critic_mappo(nn.Module):
    def __init__(self, state_size, action_size, num_agents, seed):
        super(Critic_mappo, self).__init__()
        self.fcs1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = torch.relu(self.fcs1(state))
        x = torch.relu(self.fc2(x))
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
#
# # #
# class QMIX(nn.Module):
#     def __init__(self, n_agents, state_dim):
#         super(QMIX, self).__init__()
#         self.hyper_w1 = nn.Linear(state_dim, n_agents * 256)
#         self.hyper_w2 = nn.Linear(state_dim, 256)
#         self.hyper_b1 = nn.Linear(state_dim, 256)
#         self.hyper_b2 = nn.Linear(state_dim, 1)
#
#     def forward(self, agent_qs, states):
#         batch_size = agent_qs.size(0)
#         n_agents = agent_qs.size(1)
#
#         # First layer weights and biases
#         w1 = torch.abs(self.hyper_w1(states)).view(-1, n_agents,256)
#         b1 = self.hyper_b1(states).view(-1, 1, 256)
#
#         # Second layer weights and biases
#         w2 = torch.abs(self.hyper_w2(states)).view(-1, 256, 1)
#         b2 = self.hyper_b2(states).view(-1, 1, 1)
#
#         # Ensure agent_qs is 3D
#         agent_qs = agent_qs.view(batch_size,1,n_agents)
#
#         # Mixing network
#         hidden = torch.bmm(agent_qs, w1) + b1
#         q_tot = torch.bmm(hidden, w2) + b2
#
#         return q_tot.view(batch_size, -1)
#
#






def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
