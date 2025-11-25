import numpy as np
import torch
import torch.optim as optim
from common.read_yaml import ReadYaml as read_yaml
import torch.nn.functional as F
from dev2.network import ReplayBuffer, Actor_maddpg, Critic_maddpg, Actor_mappo, Critic_mappo,DQNNet,DuelingNet

class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim,action_space):

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
        print('device:', self.device)

        model_param = param['DQN']
        self.gamma = model_param['gamma']
        self.lr = model_param['lr']
        self.gamma_discount = model_param['gamma_discount']


        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space

        # self.max_value = max_value

        # Actor and Critic Networks
        self.duelingdqn_eval_net = DuelingNet(state_dim, action_dim)
        self.duelingdqn_optimizer = optim.Adam(self.duelingdqn_eval_net.parameters(), lr=self.lr)
        self.duelingdqn_scheduler = optim.lr_scheduler.StepLR( self.duelingdqn_optimizer, self.batch_size,self.gamma)

        # Replay Buffer
        self.memory = ReplayBuffer(self.batch_size, self.batch_size)

        # loss history record
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.learn_counter = 1  # 用于记录学习次数的变量

    def test_act(self, state):
        state = state.to(torch.float32).to(self.device)
        with torch.no_grad():
            # state= state.unsqueeze(0)
            optimal_action_inx = self.duelingdqn_eval_net(state.unsqueeze(0)).max(1, keepdim=True)[1]
        optimal_action_inx = optimal_action_inx.squeeze(0).squeeze(0)
        actions = self.action_space[optimal_action_inx].squeeze(0).squeeze(0)

        return actions


    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            self.memory.clear()


    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states=states.to(self.device)
        next_states=next_states.to(self.device)
        actions=actions.to(self.device)
        dones=dones.to(self.device)
        rewards=rewards.to(self.device)

        # 计算当前状态的 Q 值
        agent_action_values = self.duelingdqn_eval_net(states)

        # # 将动作值与对应动作空间进行匹配，获取动作索引
        # 为action_space1广播,使其形状与batch_action1匹配
        # 假设 self.action_space 的形状是 [1, 23223, 2]
        action_space_expanded = self.action_space.expand(len(states), self.action_dim, 2)
        # 找到每个动作在 action_space_expanded 中的索引
        indices = []
        for i in range(actions.shape[0]):
            # 寻找匹配的索引
            matches = (action_space_expanded[i] == actions[i]).all(dim=1)
            index = torch.where(matches)[0]
            indices.append(index.item() if len(index) > 0 else -1)

        indices = torch.tensor(indices)


        # 计算当前动作对应的 Q 值
        agent_action_values = agent_action_values.gather(1, indices.unsqueeze(1))

        # # Calculate Q values for next states
        # target_action_values = self.ddqn_target_net(next_states)
        # next_action_values = self.ddqn_eval_net(next_states)
        # best_actions = next_action_values.argmax(dim=1, keepdim=True)
        # target_Q_values = target_action_values.gather(1, best_actions)


        #
        # 计算下一个状态的 Q 值
        target_action_values = self.duelingdqn_eval_net(next_states)
        target_Q_values = target_action_values.max(dim=1)[0].unsqueeze(1)

        # 根据 Bellman 方程计算目标 Q 值
        target_Q_values = rewards + self.gamma_discount * target_Q_values
        # 计算损失函数smooth_l1_loss
        loss = torch.nn.functional.mse_loss(agent_action_values, target_Q_values)

        # 更新 Critic 网络
        self.duelingdqn_optimizer.zero_grad()
        loss.backward()
        self.duelingdqn_optimizer.step()

        self.actor_loss_history.append(loss.item())

        # 每个周期更新学习率
        self.duelingdqn_scheduler.step()


    def save_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()
        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']

        torch.save(self.duelingdqn_eval_net.state_dict(), f"{weight_root}{filename}_{dataset_name}_{scheme}_bisreward.pth")

    def load_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()
        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']


        self.duelingdqn_eval_net.load_state_dict(torch.load(f"{weight_root}{filename}_{dataset_name}_{scheme}_bisreward.pth"))
        self.duelingdqn_eval_net.to(self.device)



class DDQNAgent:
    def __init__(self, state_dim, action_dim,action_space):

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
        print('device:', self.device)

        model_param = param['DDQN']
        self.gamma = model_param['gamma']
        self.lr = model_param['lr']
        self.gamma_discount = model_param['gamma_discount']
        self.target_update = model_param['target_update']


        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space

        # self.max_value = max_value

        # Actor and Critic Networks
        self.ddqn_eval_net = DQNNet(state_dim, action_dim)
        self.ddqn_target_net = DQNNet(state_dim, action_dim)
        self.ddqn_optimizer = optim.Adam(self.ddqn_eval_net.parameters(), lr=self.lr)
        self.ddqn_scheduler = optim.lr_scheduler.StepLR( self.ddqn_optimizer, self.batch_size,self.gamma)

        # Replay Buffer
        self.memory = ReplayBuffer(self.batch_size, self.batch_size)

        # loss history record
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.learn_counter = 1  # 用于记录学习次数的变量

    def test_act(self, state):
        state = state.to(torch.float32).to(self.device)
        with torch.no_grad():
            optimal_action_inx = self.ddqn_eval_net(state).max(0, keepdim=True)[1]
        actions = self.action_space[optimal_action_inx].squeeze(0)
        return actions


    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            self.memory.clear()


    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states=states.to(self.device)
        next_states=next_states.to(self.device)
        actions=actions.to(self.device)
        dones=dones.to(self.device)
        rewards=rewards.to(self.device)

        # 计算当前状态的 Q 值
        agent_action_values = self.ddqn_eval_net(states)

        # # 将动作值与对应动作空间进行匹配，获取动作索引
        # 为action_space1广播,使其形状与batch_action1匹配
        # 假设 self.action_space 的形状是 [1, 23223, 2]
        action_space_expanded = self.action_space.expand(len(states), self.action_dim, 2)
        # 找到每个动作在 action_space_expanded 中的索引
        indices = []
        for i in range(actions.shape[0]):
            # 寻找匹配的索引
            matches = (action_space_expanded[i] == actions[i]).all(dim=1)
            index = torch.where(matches)[0]
            indices.append(index.item() if len(index) > 0 else -1)

        indices = torch.tensor(indices)


        # 计算当前动作对应的 Q 值
        agent_action_values = agent_action_values.gather(1, indices.unsqueeze(1))

        # # Calculate Q values for next states
        # target_action_values = self.ddqn_target_net(next_states)
        # next_action_values = self.ddqn_eval_net(next_states)
        # best_actions = next_action_values.argmax(dim=1, keepdim=True)
        # target_Q_values = target_action_values.gather(1, best_actions)


        #
        # 计算下一个状态的 Q 值
        target_action_values = self.ddqn_target_net(next_states)
        target_Q_values = target_action_values.max(dim=1)[0].unsqueeze(1)

        # 根据 Bellman 方程计算目标 Q 值
        target_Q_values = rewards + self.gamma_discount * target_Q_values
        # 计算损失函数smooth_l1_loss
        loss = torch.nn.functional.mse_loss(agent_action_values, target_Q_values)

        # 更新 Critic 网络
        self.ddqn_optimizer.zero_grad()
        loss.backward()
        self.ddqn_optimizer.step()

        self.actor_loss_history.append(loss.item())

        # 每个周期更新学习率
        self.ddqn_scheduler.step()

        # 更新target网络
        if self.learn_counter % self.target_update == 0:
            self.ddqn_target_net.load_state_dict(self.ddqn_eval_net.state_dict())
        self.learn_counter += 1

    def save_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()
        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']

        torch.save(self.ddqn_eval_net.state_dict(), f"{weight_root}{filename}_{dataset_name}_{scheme}_bisreward.pth")

    def load_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()
        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']


        self.ddqn_eval_net.load_state_dict(torch.load(f"{weight_root}{filename}_{dataset_name}_{scheme}_bisreward.pth"))
        self.ddqn_eval_net.to(self.device)

class DQNAgent:
    def __init__(self, state_dim, action_dim,action_space):

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
        # print('ddpg_device:', self.device)

        model_param = param['DQN']
        self.gamma = model_param['gamma']
        self.lr = model_param['lr']
        self.gamma_discount = model_param['gamma_discount']


        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space

        # self.max_value = max_value

        # Actor and Critic Networks
        self.dqn = DQNNet(state_dim, action_dim)
        self.dqn_optimizer = optim.Adam(self.dqn.parameters(), self.lr)
        self.dqn_scheduler = optim.lr_scheduler.StepLR( self.dqn_optimizer, self.batch_size,self.gamma)

        # Replay Buffer
        self.memory = ReplayBuffer(self.batch_size, self.batch_size)

        # loss history record
        self.actor_loss_history = []
        self.critic_loss_history = []

    def test_act(self, state):
        state = state.to(torch.float32).to(self.device)
        with torch.no_grad():
            optimal_action_inx = self.dqn(state).max(0, keepdim=True)[1]
        actions = self.action_space[optimal_action_inx].squeeze(0)
        return actions


    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            self.memory.clear()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        states=states.to(self.device)
        next_states=next_states.to(self.device)
        actions=actions.to(self.device)
        dones=dones.to(self.device)
        rewards=rewards.to(self.device)



        # 计算当前状态的 Q 值
        agent_action_values = self.dqn(states)

        # # 将动作值与对应动作空间进行匹配，获取动作索引
        # 为action_space1广播,使其形状与batch_action1匹配
        # 假设 self.action_space 的形状是 [1, 23223, 2]
        action_space_expanded = self.action_space.expand(len(states), self.action_dim, 2)
        # 找到每个动作在 action_space_expanded 中的索引
        indices = []
        for i in range(actions.shape[0]):
            # 寻找匹配的索引
            matches = (action_space_expanded[i] == actions[i]).all(dim=1)
            index = torch.where(matches)[0]
            indices.append(index.item() if len(index) > 0 else -1)

        indices = torch.tensor(indices)


        # 计算当前动作对应的 Q 值
        agent_action_values = agent_action_values.gather(1, indices.unsqueeze(1))


        # 计算下一个状态的 Q 值
        target_action_values = self.dqn(next_states)
        target_Q_values = target_action_values.max(dim=1)[0].unsqueeze(1)

        # 根据 Bellman 方程计算目标 Q 值
        target_Q_values = rewards + self.gamma * target_Q_values
        # 计算损失函数smooth_l1_loss
        loss = torch.nn.functional.mse_loss(agent_action_values, target_Q_values)


        # #
        # action_inx = np.array([np.abs(self.action_space - action.item()).argmin() for action in actions.numpy().flatten()])
        #
        # current_q_values = self.dqn(states).gather(1, torch.tensor(action_inx).unsqueeze(1))
        #
        # # 使用target_net计算下一个状态的目标Q值
        # next_q_values = self.dqn(next_states).max(1, keepdim=True)[0]
        #
        # target_q_values = rewards + self.gamma_discount * next_q_values
        # # calculate the loss
        # loss = self.loss_func(current_q_values, target_q_values)

        # 更新 Critic 网络
        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()

        self.actor_loss_history.append(loss.item())

        # 每个周期更新学习率
        self.dqn_scheduler.step()


    def save_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()
        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']

        torch.save(self.dqn.state_dict(), f"{weight_root}{filename}_{dataset_name}_actor_{scheme}_bisreward.pth")

    def load_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()
        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']


        self.dqn.load_state_dict(torch.load(f"{weight_root}{filename}_{dataset_name}_actor_{scheme}_bisreward.pth"))
        self.dqn.to(self.device)

class DDPGAgent:
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
        # print('ddpg_device:', self.device)

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
        self.actors_local = Actor_maddpg(state_dim, action_dim,0).to(self.device)
        self.actors_target = Actor_maddpg(state_dim, action_dim,0).to(self.device)
        self.actors_optimizer = [optim.Adam(self.actors_local.parameters(), lr=self.lr_actor)]

        # 注意：这里我们假设使用一个联合 Critic 网络
        self.critic_local = Critic_maddpg(state_dim, action_dim, 0).to(self.device)
        self.critic_target = Critic_maddpg(state_dim, action_dim, 0).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Replay Buffer
        self.memory = ReplayBuffer(self.batch_size, self.batch_size)

        # loss history record
        self.actor_loss_history = []
        self.critic_loss_history = []

    def test_act(self, state):
        # 单步测试最优actions
        state = state.to(torch.float32).to(self.device)

        self.actors_local.eval()
        with torch.no_grad():
            actions = self.actors_local(state.float())
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
        # states = states.view(states.size(0), self.state_dim * self.num_agents)
        # next_states = next_states.view(-1, self.state_dim * self.num_agents)

        states=states.to(self.device)
        next_states=next_states.to(self.device)
        actions=actions.to(self.device)
        dones=dones.to(self.device)
        rewards=rewards.to(self.device)

        # states = torch.cat([states, states], dim=1)
        # next_states = torch.cat([next_states, next_states], dim=1)
        # states = states.view(states.size(0), self.state_dim * self.num_agents).to(self.device)
        # next_states = next_states.view(-1, self.state_dim * self.num_agents).to(self.device)
        # actions = actions.view(-1, self.action_dim * self.num_agents).to(self.device)
        # dones = dones.view(-1, 1).to(self.device)
        # rewards = rewards.view(-1, 1).to(self.device)

        # 计算下一个状态的动作
        actions_next = self.actors_target(next_states)

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

        # 更新 Actor 网络.mean()
        actions_pred = self.actors_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        for actor_optimizer in self.actors_optimizer:
            actor_optimizer.zero_grad()
        actor_loss.backward()
        for actor_optimizer in self.actors_optimizer:
            actor_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.critic_local, self.critic_target, self.tau)

        self.soft_update(self.actors_local, self.actors_target, self.tau)

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

        torch.save(self.actors_local.state_dict(), f"{weight_root}{filename}_{dataset_name}_actor_{scheme}_bisreward.pth")

        torch.save(self.critic_local.state_dict(), f"{weight_root}{filename}_{dataset_name}_critic_{scheme}_bisreward.pth")

    def load_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()
        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']


        self.actors_local.load_state_dict(torch.load(f"{weight_root}{filename}_{dataset_name}_actor_{scheme}_bisreward.pth"))
        self.actors_local.to(self.device)

        self.critic_local.load_state_dict(torch.load(f"{weight_root}{filename}_{dataset_name}_critic_{scheme}_bisreward.pth"))
        self.critic_local.to(self.device)


class PPOAgent:
    def __init__(self, state_dim, action_dim,max_value):
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

        model_param = param['MAPPO']
        self.gamma = model_param['gamma']
        self.tau = model_param['tau']
        self.lr_actor = model_param['lr_actor']
        self.lr_critic = model_param['lr_critic']
        self.weight_decay = model_param['weight_decay']
        self.epsilon = model_param['epsilon']  # PPO clip parameter
        self.epochs = model_param['epochs']  # Number of epochs for PPO update
        self.clip_param = model_param['clip_param']  # PPO clip parameter

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_value = max_value

        # Actor Networks (one for each agent)
        self.actors = Actor_mappo(state_dim, action_dim, 0,self.max_value).to(self.device)
        self.actors_optimizer = optim.Adam(self.actors.parameters(), lr=self.lr_actor)

        # Centralized Critic Network
        self.critic = Critic_mappo(state_dim, action_dim, self.num_agents, 0).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Replay Buffer
        self.memory = ReplayBuffer(self.batch_size, self.batch_size)

        # Loss history record
        self.actor_loss_history = []
        self.critic_loss_history = []

    def test_act(self, state):
        # 单步测试最优actions
        state = state.to(torch.float32).to(self.device)

        self.actors.eval()
        with torch.no_grad():
            actions = self.actors(state.float())
            actions = torch.tensor(actions)
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            self.memory.clear()

    def learn(self, experiences):
        observations, actions, rewards, next_observations, dones = experiences

        observations = observations.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_observations = next_observations.to(self.device)
        dones = dones.to(self.device)

        # Reshape observations for the centralized critic
        obs_critic = observations.view(observations.size(0), -1)
        next_obs_critic = next_observations.view(next_observations.size(0), -1)

        # Compute value estimates
        with torch.no_grad():
            values = self.critic(obs_critic)
            next_values = self.critic(next_obs_critic)

        # Compute advantages
        advantages = self.compute_gae(rewards, values, next_values, dones)


        new_actions = self.actors(observations)
        # Compute the probability ratio
        ratio = self.compute_ratio(new_actions, actions.unsqueeze(1), observations)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.5 - self.clip_param, 0.5 + self.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Update actor networks

        self.actors_optimizer.zero_grad()
        actor_loss.backward()
        self.actors_optimizer.step()
        # Update critic
        value_pred = self.critic(obs_critic)
        value_target = rewards + self.gamma * next_values * (1 - dones)
        value_loss = F.mse_loss(value_pred, value_target.detach())

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(value_loss.item())

    def compute_ratio(self, new_actions, old_actions, states):
        new_log_probs = self.log_prob(new_actions, states)
        old_actions=old_actions.unsqueeze(1)
        old_log_probs = self.log_prob(old_actions, states)
        return torch.exp(new_log_probs - old_log_probs)

    def log_prob(self, actions, states):
        # 假设动作是在 tanh 空间中的，我们需要将其转换回正态分布空间
        # actions = torch.clamp(actions, 0 + 1e-6, 1 - 1e-6)
        # actions = torch.clamp(actions, 0, 0.0338)
        # actions = 0.5 * torch.log((1 + actions) / (1 - actions))

        # 计算均值和标准差
        mean = self.actors(states)
        log_std = self.actors.log_std.expand_as(mean)
        std = torch.exp(log_std)

        # 计算动作的对数概率
        log_probs = -((actions - mean) ** 2) / (2 * std.pow(2)) - \
                    log_std - np.log(np.sqrt(2 * np.pi))

        # 求和得到总的对数概率
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        # # 应用 tanh 变换的修正项
        # log_probs -= torch.sum(torch.log(1 - actions.pow(2) + 1e-6), dim=-1, keepdim=True)

        return log_probs

    def compute_gae(self, rewards, values, next_values, dones):
        gae = 0
        advantages = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.tau * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        return advantages

    def save_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()

        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']



        torch.save(self.actors.state_dict(), f"{weight_root}{filename}_{dataset_name}_actor_{scheme}_bisreward.pth")

        torch.save(self.critic.state_dict(), f"{weight_root}{filename}_{dataset_name}_critic_{scheme}_bisreward.pth")

    def load_model(self, filename, scheme,dataset_name):
        param = read_yaml()
        param = param.read()

        basic_config = param['basic_config']
        weight_root = basic_config['weight_root']


        self.actors.load_state_dict(torch.load(f"{weight_root}{filename}_{dataset_name}_actor_{scheme}_bisreward.pth"))
        self.actors.to(self.device)

        self.critic.load_state_dict(torch.load(f"{weight_root}{filename}_{dataset_name}_critic_{scheme}_bisreward.pth"))
        self.critic.to(self.device)
