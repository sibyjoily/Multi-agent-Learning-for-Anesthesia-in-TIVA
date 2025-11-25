import numpy as np
import torch
import torch.optim as optim
from common.read_yaml import ReadYaml as read_yaml
import torch.nn.functional as F
from dev2.network import ReplayBuffer, Actor_maddpg, Critic_maddpg, Actor_coma, Critic_coma, \
    Actor_mappo, Critic_mappo, VDNNet, qmixNet,QMIX,qplexNet,DuelingMixingNetwork,CW_qmixNet,\
    compute_cw_weights,CW_QMIXNet


class MADDPGAgent:
    def __init__(self, state_dim, action_dim):

        param = read_yaml()
        param = param.read()

        model_common = param['model_common']
        self.max_episodes = model_common['max_episodes']
        self.tolerance = model_common['tolerance']
        self.loss_threshold = model_common['loss_threshold']
        self.batch_size = model_common['batch_size']
        self.num_agents = model_common['num_agents']
        self.device = model_common['device']
        if self.device == 'mac':
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif self.device == 'windows':
            self.device = torch.device("cpu")
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print('device:', self.device)

        model_param = param['MADDPG']
        self.gamma = model_param['gamma']
        self.tau = model_param['tau']
        self.lr_actor = model_param['lr_actor']
        self.lr_critic = model_param['lr_critic']
        self.weight_decay = model_param['weight_decay']

        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.max_value = max_value

        # Actor and Critic Networks
        self.actors_local = [Actor_maddpg(state_dim, action_dim, i).to(self.device) for i in range(self.num_agents)]
        self.actors_target = [Actor_maddpg(state_dim, action_dim, i).to(self.device) for i in range(self.num_agents)]
        self.actors_optimizer = [optim.Adam(actor.parameters(), lr=self.lr_actor) for actor in self.actors_local]

        # 注意：这里我们假设使用一个联合 Critic 网络
        self.critic_local = Critic_maddpg(state_dim * self.num_agents, action_dim * self.num_agents, 0).to(self.device)
        self.critic_target = Critic_maddpg(state_dim * self.num_agents, action_dim * self.num_agents, 0).to(self.device)


        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Replay Buffer
        self.memory = ReplayBuffer(self.batch_size, self.batch_size)

        # loss history record
        self.actor_loss_history = []
        self.critic_loss_history = []

    def test_act(self, state):
        # 单步测试最优actions
        state = state.to(torch.float32).to(self.device)
        actions = []
        for i in range(self.num_agents):
            self.actors_local[i].eval()
            # state = state.to(self.device)
            # self.actors_local[i] = self.actors_local[i].to('cpu')
            with torch.no_grad():
                action = self.actors_local[i](state.float())
                actions.append(action.item())
        actions = torch.tensor(actions)
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            self.memory.clear()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # 将所有智能体的状态和动作拼接

        states = torch.cat([states, states], dim=1)
        next_states = torch.cat([next_states, next_states], dim=1)
        states = states.view(states.size(0), self.state_dim * self.num_agents).to(self.device)
        next_states = next_states.view(-1, self.state_dim * self.num_agents).to(self.device)


        actions = actions.view(-1, self.action_dim * self.num_agents).to(self.device)
        dones = dones.view(-1, 1).to(self.device)
        rewards = rewards.view(-1, 1).to(self.device)

        # 计算下一个状态的动作
        actions_next = torch.cat([self.actors_target[j](next_states[:, j * self.state_dim:(j + 1) * self.state_dim]) for j in range(self.num_agents)], dim=1)
        # 计算目标 Q 值
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards.view(-1, 1) + (self.gamma * Q_targets_next * (1 - dones.view(-1, 1)))

        # 计算当前 Q 值
        Q_expected = self.critic_local(states, actions)

        # 计算损失
        critic_loss = torch.nn.functional.mse_loss(Q_expected, Q_targets)

        # 更新 Critic 网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # 更新 Actor 网络
        actions_pred = torch.cat([self.actors_local[j](states[:, j * self.state_dim:(j + 1) * self.state_dim]) for j in range(self.num_agents)], dim=1)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        for actor_optimizer in self.actors_optimizer:
            actor_optimizer.zero_grad()
        actor_loss.backward()
        for actor_optimizer in self.actors_optimizer:
            actor_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        for i in range(self.num_agents):
            self.soft_update(self.actors_local[i], self.actors_target[i], self.tau)

        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()

        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']

        for i, actor in enumerate(self.actors_local):
            torch.save(actor.state_dict(), f"{weight_root}{filename}_{dataset_name}_actor_{scheme}_{i}_bisreward.pth")

        torch.save(self.critic_local.state_dict(), f"{weight_root}{filename}_{dataset_name}_critic_{scheme}_{i}_bisreward.pth")

    def load_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()

        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']

        for i, actor in enumerate(self.actors_local):
            actor.load_state_dict(torch.load(f"{weight_root}{filename}_{dataset_name}_actor_{scheme}_{i}_bisreward.pth"))
            actor.to(self.device)

        self.critic_local.load_state_dict(torch.load(f"{weight_root}{filename}_{dataset_name}_critic_{scheme}_{i}_bisreward.pth"))
        self.critic_local.to(self.device)

class COMAAgent:
    def __init__(self, state_dim, action_dim):

        param = read_yaml()
        param = param.read()

        model_common = param['model_common']
        self.max_episodes = model_common['max_episodes']
        self.tolerance = model_common['tolerance']
        self.loss_threshold = model_common['loss_threshold']
        self.batch_size = model_common['batch_size']
        self.num_agents = model_common['num_agents']
        self.device = model_common['device']
        if self.device == 'mac':
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif self.device == 'windows':
            self.device = torch.device("cpu")
            # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_param = param['COMA']
        self.gamma = model_param['gamma']
        self.lr_actor = model_param['lr_actor']
        self.lr_critic = model_param['lr_critic']
        self.weight_decay = model_param['weight_decay']

        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.max_value = max_value
        # Actor Networks
        self.actors = [Actor_coma(state_dim, action_dim, i).to(self.device) for i in range(self.num_agents)]
        self.actors_optimizer = [optim.Adam(actor.parameters(), lr=self.lr_actor) for actor in self.actors]

        # Centralized Critic Network
        self.critic = Critic_coma(state_dim, action_dim * self.num_agents, 0).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Replay Buffer
        self.memory = ReplayBuffer(self.batch_size, self.batch_size)

        # loss history record
        self.actor_loss_history = []
        self.critic_loss_history = []

    def test_act(self, state):
        # 单步测试最优actions
        state = state.to(torch.float32).to(self.device)
        actions = []
        for i in range(self.num_agents):
            self.actors[i].eval()
            with torch.no_grad():
                action = self.actors[i](state.float())
                actions.append(action.item())
        actions = torch.tensor(actions)
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            self.memory.clear()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Reshape for multi-agent
        states = states.view(states.size(0), self.state_dim).to(self.device)
        next_states = next_states.view(-1, self.state_dim).to(self.device)
        actions = actions.view(-1, self.action_dim * self.num_agents).to(self.device)
        dones = dones.view(-1, 1).to(self.device)
        rewards = rewards.view(-1, 1).to(self.device)

        # Calculate target Q values
        Q_targets_next = self.critic(next_states, actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Calculate expected Q values
        Q_expected = self.critic(states, actions)

        # Calculate critic loss
        critic_loss = torch.nn.functional.mse_loss(Q_expected, Q_targets)

        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Calculate actor loss
        actor_loss = 0
        for i in range(self.num_agents):
            # actions_pred = self.actors[i](stactions_pred = self.actors[i](states)ates[:, i * self.state_dim:(i + 1) * self.state_dim])
            actions_pred = self.actors[i](states)
            actions_all = actions.clone()
            actions_all[:, i * self.action_dim:(i + 1) * self.action_dim] = actions_pred
            actor_loss -= self.critic(states, actions_all).mean()

        # Update actor networks
        for actor_optimizer in self.actors_optimizer:
            actor_optimizer.zero_grad()
        actor_loss.backward()
        for actor_optimizer in self.actors_optimizer:
            actor_optimizer.step()

        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())

    def save_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()

        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']

        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), f"{weight_root}{filename}_{dataset_name}_actor_{scheme}_{i}_bisreward.pth")

        torch.save(self.critic.state_dict(), f"{weight_root}{filename}_{dataset_name}_critic_{scheme}_bisreward.pth")

    def load_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()

        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']

        for i, actor in enumerate(self.actors):
            actor.load_state_dict(torch.load(f"{weight_root}{filename}_{dataset_name}_actor_{scheme}_{i}_bisreward.pth"))
            actor.to(self.device)

        self.critic.load_state_dict(torch.load(f"{weight_root}{filename}_{dataset_name}_critic_{scheme}_bisreward.pth"))
        self.critic.to(self.device)

class VDNAgent:
    def __init__(self, state_dim, action_dim1,action_dim2,action_space1,action_space2):

        param = read_yaml()
        param = param.read()

        model_common = param['model_common']
        self.max_episodes = model_common['max_episodes']
        self.tolerance = model_common['tolerance']
        self.loss_threshold = model_common['loss_threshold']
        self.batch_size = model_common['batch_size']
        self.num_agents = model_common['num_agents']
        self.device = model_common['device']
        if self.device == 'mac':
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif self.device == 'windows':
            self.device = torch.device("cpu")
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print('device:', self.device)

        model_param = param['VDN']
        self.gamma = model_param['gamma']
        self.lr = model_param['lr']
        self.gamma_discount = model_param['gamma_discount']


        self.state_dim = state_dim
        self.action_dim1 = action_dim1
        self.action_dim2 = action_dim2
        self.action_space1 = action_space1
        self.action_space2 = action_space2
        # self.max_value = max_value

        # 初始化网络
        self.VDNnet = VDNNet(self.state_dim, self.action_dim1, self.action_dim2).to(self.device)
        self.VDNnet_optimizer = optim.Adam(self.VDNnet.parameters(), self.lr)
        self.VDNnet_scheduler = optim.lr_scheduler.StepLR(self.VDNnet_optimizer, self.batch_size, self.gamma)

        # Replay Buffer
        self.memory = ReplayBuffer(self.batch_size, self.batch_size)

        # loss history record
        self.actor_loss_history = []
        self.critic_loss_history = []

    def test_act(self, state):
        # 单步测试最优actions
        state = state.to(torch.float32).to(self.device)
        # 使用eval_net选择动作
        with torch.no_grad():
            action_values1, action_values2 = self.VDNnet(state)

        # 根据动作值选择动作的索引
        action_idx1 = torch.argmax(action_values1).item()
        action_idx2 = torch.argmax(action_values2).item()
        # 获取索引在动作空间中对应的动作
        action1 = self.action_space1[action_idx1]
        action2 = self.action_space2[action_idx2]
        #将action1和action2按列拼接
        action1 = action1.unsqueeze(0)
        action2 = action2.unsqueeze(0)
        actions = torch.cat([action1, action2], dim=0)
        return actions


    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            self.memory.clear()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Reshape for multi-agent
        states = states.view(-1, self.state_dim).to(self.device)
        next_states = next_states.view(-1, self.state_dim).to(self.device)
        actions = actions.view(-1, 2).to(self.device)
        batch_action1 = actions[:, 0].unsqueeze(1)
        batch_action2 = actions[:, 1].unsqueeze(1)
        dones = dones.view(-1, 1).to(self.device)
        rewards = rewards.view(-1, 1).to(self.device)

        # 计算当前状态的 Q 值
        agent1_action_values, agent2_action_values = self.VDNnet(states)

        # # 将动作值与对应动作空间进行匹配，获取动作索引
        # 为action_space1广播,使其形状与batch_action1匹配
        action_space1_expanded = self.action_space1.unsqueeze(0).expand(len(states), -1)
        # 为action_space2广播,使其形状与batch_action2匹配
        action_space2_expanded = self.action_space2.unsqueeze(0).expand(len(states), -1)

        batch_action1_indices = torch.searchsorted(action_space1_expanded, batch_action1)
        batch_action2_indices = torch.searchsorted(action_space2_expanded, batch_action2)

        # 计算当前动作对应的 Q 值
        agent1_action_values = agent1_action_values.gather(1, batch_action1_indices)
        agent2_action_values = agent2_action_values.gather(1, batch_action2_indices)

        Q_tot_values = agent1_action_values + agent2_action_values

        # 计算下一个状态的 Q 值
        target_action_values1, target_action_values2 = self.VDNnet(next_states)
        target_Q_tot_values = target_action_values1.max(dim=1)[0].unsqueeze(1) + target_action_values2.max(dim=1)[
            0].unsqueeze(1)

        # 根据 Bellman 方程计算目标 Q 值
        target_Q_tot_values = rewards + self.gamma * target_Q_tot_values
        # 计算损失函数smooth_l1_loss
        total_loss = torch.nn.functional.mse_loss(Q_tot_values, target_Q_tot_values)

        # 更新 Critic 网络
        self.VDNnet_optimizer.zero_grad()
        total_loss.backward()
        self.VDNnet_optimizer.step()

        self.actor_loss_history.append(total_loss.item())

        # 每个周期更新学习率
        self.VDNnet_scheduler.step()

    def save_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()

        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']

        torch.save(self.VDNnet.state_dict(), f"{weight_root}{filename}_{dataset_name}_{scheme}_bisreward.pth")

    def load_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()

        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']

        self.VDNnet.load_state_dict(torch.load(f"{weight_root}{filename}_{dataset_name}_{scheme}_bisreward.pth"))
        self.VDNnet.to(self.device)

class QMIXAgent:
    def __init__(self, state_dim, action_dim1,action_dim2,action_space1,action_space2):

        param = read_yaml()
        param = param.read()

        model_common = param['model_common']
        self.max_episodes = model_common['max_episodes']
        self.tolerance = model_common['tolerance']
        self.loss_threshold = model_common['loss_threshold']
        self.batch_size = model_common['batch_size']
        self.num_agents = model_common['num_agents']
        self.device = model_common['device']
        if self.device == 'mac':
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif self.device == 'windows':
            self.device = torch.device("cpu")
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print('device:', self.device)

        model_param = param['QMIX']
        self.gamma = model_param['gamma']
        self.lr = model_param['lr']
        self.gamma_discount = model_param['gamma_discount']



        self.state_dim = state_dim
        self.action_dim1 = action_dim1
        self.action_dim2 = action_dim2
        self.action_space1 = action_space1
        self.action_space2 = action_space2
        # self.max_value = max_value

        # 初始化网络
        # 创建 QMIX 网络实例
        self.qmix_eval_net = qmixNet(self.state_dim, self.action_dim1, self.action_dim2)
        # self.qmix_target_net = qmixNet(self.state_dim, self.action_dim1, self.action_dim2)
        self.mixer = QMIX(self.num_agents,self.state_dim)

        self.qmix_optimizer = torch.optim.Adam(list(self.qmix_eval_net.parameters()) + list(self.mixer.parameters()), lr=self.lr)

        self.qmix_scheduler = optim.lr_scheduler.StepLR(self.qmix_optimizer, self.batch_size, self.gamma)

        # Replay Buffer
        self.memory = ReplayBuffer(self.batch_size, self.batch_size)

        # loss history record
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.learn_counter = 1  # 用于记录学习次数的变量

    def test_act(self, state):
        # 单步测试最优actions
        state = state.to(torch.float32).to(self.device)
        # 使用eval_net选择动作
        with torch.no_grad():
            action_values1, action_values2 = self.qmix_eval_net(state)

        # 根据动作值选择动作的索引
        action_idx1 = torch.argmax(action_values1).item()
        action_idx2 = torch.argmax(action_values2).item()
        # 获取索引在动作空间中对应的动作
        action1 = self.action_space1[action_idx1]
        action2 = self.action_space2[action_idx2]
        #将action1和action2按列拼接
        action1 = action1.unsqueeze(0)
        action2 = action2.unsqueeze(0)
        actions = torch.cat([action1, action2], dim=0)
        return actions

    # def step(self, states, actions, rewards, next_states, dones):
    #     self.memory.add(states, actions, rewards, next_states, dones)
    #     if dones== True:
    #         experiences = self.memory.sample()
    #         self.learn(experiences)
    #         self.memory.clear()
    #     # 更新target网络
    #     if self.learn_counter % self.target_update == 0:
    #         self.qmix_target_net.load_state_dict(self.qmix_eval_net.state_dict())
    #     self.learn_counter += 1
    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)

        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            self.memory.clear()



    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Reshape for multi-agent
        states = states.view(-1, self.state_dim).to(self.device)
        next_states = next_states.view(-1, self.state_dim).to(self.device)
        actions = actions.view(-1, 2).to(self.device)
        batch_action1 = actions[:, 0].unsqueeze(1)
        batch_action2 = actions[:, 1].unsqueeze(1)
        dones = dones.view(-1, 1).to(self.device)
        rewards = rewards.view(-1, 1).to(self.device)
        # 计算当前状态的 Q 值
        agent1_action_values, agent2_action_values = self.qmix_eval_net(states)

        # # 将动作值与对应动作空间进行匹配，获取动作索引
        # 为action_space1广播,使其形状与batch_action1匹配
        action_space1_expanded = self.action_space1.unsqueeze(0).expand(len(states), -1)
        # 为action_space2广播,使其形状与batch_action2匹配
        action_space2_expanded = self.action_space2.unsqueeze(0).expand(len(states), -1)

        batch_action1_indices = torch.searchsorted(action_space1_expanded, batch_action1)
        batch_action2_indices = torch.searchsorted(action_space2_expanded, batch_action2)

        # 计算当前动作对应的 Q 值
        agent1_action_values = agent1_action_values.gather(1, batch_action1_indices)
        agent2_action_values = agent2_action_values.gather(1, batch_action2_indices)

        cur_agent_total = torch.stack([agent1_action_values, agent2_action_values], dim=1).squeeze(2)
        current_q_tot = self.mixer(cur_agent_total, states)
        # 使用target_net计算下一个状态的目标Q值
        next_q_values1, next_q_values2 = self.qmix_eval_net(next_states)

        next_q_values1_max = next_q_values1.max(1, keepdim=True)[0]
        next_q_values2_max = next_q_values2.max(1, keepdim=True)[0]

        next_agent_total = torch.stack([next_q_values1_max, next_q_values2_max], dim=1).squeeze(2)
        next_q_tot = self.mixer(next_agent_total, next_states)

        # 计算分解的Q值
        # target_q_values = self.mixer(batch_state)  # 使用混合网络计算全局Q值
        target_q_tot = rewards + self.gamma_discount * next_q_tot

        # 计算损失函数
        total_loss = torch.nn.functional.mse_loss(current_q_tot, target_q_tot)

        # 更新 Critic 网络
        self.qmix_optimizer.zero_grad()
        total_loss.backward()
        self.qmix_optimizer.step()

        self.actor_loss_history.append(total_loss.item())

        # 每个周期更新学习率
        self.qmix_scheduler.step()

        # # 更新target网络
        # if self.learn_counter % self.target_update == 0:
        #     self.qmix_target_net.load_state_dict(self.qmix_eval_net.state_dict())
        # self.learn_counter += 1

    def save_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()

        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']

        torch.save(self.qmix_eval_net.state_dict(), f"{weight_root}{filename}_{dataset_name}_{scheme}_bisreward.pth")

    def load_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()

        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']

        self.qmix_eval_net.load_state_dict(torch.load(f"{weight_root}{filename}_{dataset_name}_{scheme}_bisreward.pth"))
        self.qmix_eval_net.to(self.device)
