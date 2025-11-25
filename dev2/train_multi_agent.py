import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from common.data_processing_test import proceesing
from dev2.multi_agent_model import MADDPGAgent, COMAAgent, MAPPOAgent,VDNAgent,QMIXAgent,QPLEXAgent,CW_QMIXAgent
from common.read_yaml import ReadYaml as read_yaml
import os
from tqdm import tqdm
import pandas as pd
import joblib
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

class MultiAgentPolicyBased:
    def __init__(self, model_name='MADDPG', dataset_name='general', is_sampling=True, scheme='online',
                 test_env_model_name='gbr_general_1',):
        '''

        :param model_name: MARL的模型名称，包含maddpg
        :param dataset_name: 数据集是general还是thoracic
        :param is_sampling: 是否要只取几个case从数据集中，用于测试。少量case跑的更快、更容易debug
        '''

        self.model_name = model_name
        self.scheme = scheme
        self.dataset_name = dataset_name

        # basic configures
        param = read_yaml()
        param = param.read()
        basic_param = param['basic_config']
        self.result_root = basic_param['result_root']
        self.image_root = basic_param['image_root']
        self.weight_root = basic_param['weight_root']
        self.loss_image_root = basic_param['loss_image_root']
        self.test_root = basic_param['test_root']
        self.predata_root = basic_param['predata_root']
        model_common = param['model_common']
        self.max_episodes = model_common['max_episodes']
        self.epsilon = model_common['epsilon']
        platform = model_common['device']

        addresses = [self.result_root, self.image_root, self.weight_root, self.loss_image_root]

        for address in addresses:
            if not os.path.exists(address):
                # 如果地址不存在，则创建目录
                os.makedirs(address)

        # common model configures
        common_param = param['model_common']
        self.num_agents = common_param['num_agents']

        # other configures
        if platform == 'mac':
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif platform == 'windows':
            # self.device = torch.device("cpu")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('device:', self.device)

        # abstract the dataset and calculate the dimension
        process = proceesing()
        if self.dataset_name == 'general':
            states_dl, actions, next_state_dl, rewards, case_ids = process.experiment_data_general_agenttrain(is_sampling)
            agent_action_space = process.experiment_data_all(train_test='general')
        elif self.dataset_name == 'thoracic':
            states_dl, actions, next_state_dl, rewards, case_ids = process.experiment_data_thoracic_agenttrain(is_sampling)
            agent_action_space = process.experiment_data_all(train_test='thoracic')

        self.actions = actions
        self.states = states_dl
        self.next_states = next_state_dl
        self.rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        self.case_ids = torch.tensor(case_ids).unsqueeze(1)
        self.unique_caseids = np.unique(case_ids)

        # 计算 actions 的最大值
        self.max_value = torch.max(actions, dim=0).values

        self.state_dim = np.shape(self.states)[1]
        self.action_dim = int(np.shape(self.actions)[1] / self.num_agents)
        self.sample_num = np.shape(self.states)[0]

        self.action_dim1 = agent_action_space[:, 0].unique().shape[0]
        self.action_dim2 = agent_action_space[:, 1].unique().shape[0]
        # self.action_tot_dim=1
        self.action_space1 = agent_action_space[:, 0].unique()
        self.action_space2 = agent_action_space[:, 1].unique()

        # load the environment model
        self.env_model = joblib.load(self.weight_root + f'{test_env_model_name}.pkl')

        if self.model_name == 'MADDPG':
            self.model = MADDPGAgent(state_dim=self.state_dim, action_dim=self.action_dim)
        elif self.model_name == 'COMA':
            self.model = COMAAgent(state_dim=self.state_dim, action_dim=self.action_dim)
        elif self.model_name == 'MAPPO':
            self.model = MAPPOAgent(state_dim=self.state_dim, action_dim=self.action_dim,max_value=self.max_value)
        elif self.model_name == 'VDN':
            self.model = VDNAgent(state_dim=self.state_dim, action_dim1=self.action_dim1,
                                  action_dim2=self.action_dim2, action_space1=self.action_space1,
                                  action_space2=self.action_space2)
        elif self.model_name == 'QMIX':
            self.model = QMIXAgent(state_dim=self.state_dim, action_dim1=self.action_dim1,
                                  action_dim2=self.action_dim2, action_space1=self.action_space1,
                                  action_space2=self.action_space2)
        elif self.model_name == 'QPLEX':
            self.model = QPLEXAgent(state_dim=self.state_dim, action_dim1=self.action_dim1,
                                   action_dim2=self.action_dim2, action_space1=self.action_space1,
                                   action_space2=self.action_space2)
        elif self.model_name == 'CW_QMIX':
            self.model = CW_QMIXAgent(state_dim=self.state_dim, action_dim1=self.action_dim1,
                                    action_dim2=self.action_dim2,
                                    action_space1=self.action_space1,
                                    action_space2=self.action_space2)


    def step_a_episode(self):
        for i in range(len(self.unique_caseids)):
            caseid = self.unique_caseids[i]
            first_index = np.where(self.case_ids == caseid)[0][0]
            last_index = np.where(self.case_ids == caseid)[0][-1]
            states = self.states[first_index:last_index + 1]

            if self.scheme == 'offline':
                actions = self.actions[first_index:last_index + 1]
                rewards = self.rewards[first_index:last_index + 1]
                next_states = self.next_states[first_index:last_index + 1]
            elif self.scheme == 'online':
                state = states[0]
                agent_input = state
                state_demographic = state.squeeze()[:4].detach().tolist()
                state = state[:-self.num_agents]
                action_cum = np.array([0] * self.num_agents)
            # for j in tqdm(range(last_index - first_index + 1)):
            for j in range(last_index - first_index + 1):
                if j == last_index - first_index + 1:
                    is_done = True
                else:
                    is_done = False

                if self.scheme == 'offline':
                    self.model.step(states[j], actions[j], rewards[j], next_states[j], is_done)
                elif self.scheme == 'online':
                    actions = self.model.test_act(agent_input)
                    action_cum = action_cum + np.array(actions.tolist())
                    env_input = torch.cat((state, actions), 0)
                    env_input = np.array(env_input).reshape(1, -1)

                    output = self.env_model.predict(env_input)
                    next_state, reward, bis = self.reformat_env_model_output(output, state_demographic, env_input,actions[0],actions[1])

                    self.model.step(agent_input, actions, reward, torch.cat((next_state, torch.tensor(action_cum)), dim=0), is_done)

                    # self.model.step(agent_input, actions, reward, torch.cat(next_state, torch.tensor(action_cum)), is_done)

                    state = next_state
                    agent_input = torch.cat((state, torch.tensor(action_cum)), dim=0)

    def run(self):
        for episode in tqdm(range(self.max_episodes)):
            self.step_a_episode()


    # def act_for_online(self, state):
    #     if random.random() > self.epsilon:
    #         actions = np.random.rand(self.num_agents)
    #     else:
    #         actions = self.model.act(self.state)

    def save(self):
        self.model.save_model(self.model_name, self.scheme,self.dataset_name)
        print('model saved')

    def draw(self):
        # 画出loss曲线
        fig, ax = plt.subplots(figsize=(8, 5))
        # plt.figure(figsize=(10, 5))
        # plt.plot(loss_history[100:])
        plt.plot(self.model.actor_loss_history)
        plt.plot(self.model.critic_loss_history)
        plt.xlabel('Iteration', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title(f'{self.model_name} {self.scheme} Loss', fontsize=16)
        # 修改坐标轴刻度的大小
        ax.tick_params(axis='x', labelsize=14)  # 修改 x 轴刻度的大小
        ax.tick_params(axis='y', labelsize=14)  # 修改 y 轴刻度的大小

        plt.savefig(self.loss_image_root + f'{self.model_name}_{self.dataset_name}_{self.scheme}.pdf', format='pdf', bbox_inches='tight')
        print('loss image saved')
        plt.show()

    def data_type_convert(self,trajectory_predict_data):
        trajectory_predict_data['state'] = list(
            map(lambda x: x.detach().numpy().squeeze(), trajectory_predict_data['state']))
        trajectory_predict_data['state'] = list(map(lambda x: x.tolist(), trajectory_predict_data['state']))
        trajectory_predict_data['actions'] = list(
            map(lambda x: x.detach().numpy().squeeze(), trajectory_predict_data['actions']))
        trajectory_predict_data['actions'] = list(
            map(lambda x: x.tolist(), trajectory_predict_data['actions']))
        # trajectory_predict_data['action_single']=list(map(lambda x:x.tolist(),trajectory_predict_data['action_single']))
        trajectory_predict_data['next_state'] = list(
            map(lambda x: x.detach().numpy().squeeze(), trajectory_predict_data['next_state']))
        trajectory_predict_data['next_state'] = list(map(lambda x: x.tolist(), trajectory_predict_data['next_state']))
        trajectory_predict_data['caseid'] = list(map(lambda x: x.tolist(), trajectory_predict_data['caseid']))
        return trajectory_predict_data

    def test(self, reload=False,is_sampling=True,dataset_name='general'):
        ############################################################################################################
        process = proceesing()
        if dataset_name == 'general':
            states_dl, actions, next_state_dl, rewards, case_ids = process.experiment_data_general_agenttest(is_sampling)
        elif dataset_name == 'thoracic':
            states_dl, actions, next_state_dl, rewards, case_ids = process.experiment_data_thoracic_agenttest(is_sampling)

        # self.actions = actions
        states = states_dl
        # self.next_states = next_state_dl
        # self.rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        case_ids = torch.tensor(case_ids).unsqueeze(1)
        # self.unique_caseids = np.unique(case_ids)
        ############################################################################################################
        # load the agent model
        if reload==True:
            self.model.load_model(self.model_name, self.scheme,self.dataset_name)
        # result = pd.DataFrame(columns=['state', 'actions', 'reward', 'next_state', 'bis', 'caseid', 'step'])
        result = {'state': [], 'actions': [], 'reward': [], 'next_state': [], 'bis': [], 'caseid': [], 'step': []}
        # generate the initial state for every case
        # caseid_unique_list = list(set(case_ids)) # 从case_ids中提取出唯一的caseid# 从case_ids中提取出唯一的caseid
        # caseid_unique_list = list(map(lambda x: x.item(), caseid_unique_list)) # 将tensor转换为int
        # cased_id_list = list(map(lambda x: x.item(), case_ids))
        caseid_unique_list = torch.unique(case_ids)
        cased_id_list = list(map(lambda x: x.item(), case_ids))
        # 将tensor转换为int
        caseid_unique_list_first_state = list(map(lambda x: states[cased_id_list.index(x)], caseid_unique_list)) # 用于存储每个caseid的第一个state

        # 测试
        # caseid_unique_list = caseid_unique_list[0:50]

        # for i in tqdm(range(len(caseid_unique_list))):
        for i in range(len(caseid_unique_list)):
            # agent的state后面的两位action累加、原始数据的state后两位action也是累加的，env model输入的state后两位action不累加，env model输出的state没有action和state_demographic
            action_cum = np.array([0] * self.num_agents)
            all_steps = cased_id_list.count(caseid_unique_list[i])
            state = caseid_unique_list_first_state[i]
            agent_input = state
            state_demographic = state.squeeze()[:4].detach().tolist()
            state = state[:-self.num_agents]

            for j in range(1, all_steps):
                actions = self.model.test_act(agent_input)
                action_cum = action_cum + np.array(actions.tolist())
                env_input = torch.cat((state, actions), 0)
                env_input = np.array(env_input).reshape(1, -1)

                output = self.env_model.predict(env_input)
                next_state, reward, bis  = self.reformat_env_model_output(output, state_demographic, env_input,actions[0],actions[1])

                result['state'].append(state)
                result['actions'].append(actions)
                result['reward'].append(reward)
                result['next_state'].append(next_state)
                result['bis'].append(bis)
                result['caseid'].append(caseid_unique_list[i])
                result['step'].append(j)

                state = next_state
                agent_input = torch.cat((state, torch.tensor(action_cum)), 0)

        result_df = pd.DataFrame(result)
        #
        result_df = self.data_type_convert(result_df)

        result_df.to_csv(self.test_root + f'test_df_{self.scheme}_{dataset_name}_{self.model_name}_bisreward.csv', index=False)
        print('test result saved')

    def test_fixaction(self, reload=False,is_sampling=True,dataset_name='general'):
        ############################################################################################################
        process = proceesing()
        if dataset_name == 'general':
            states_dl, actions, next_state_dl, rewards, case_ids = process.experiment_data_general_agenttest(is_sampling)
        elif dataset_name == 'thoracic':
            states_dl, actions, next_state_dl, rewards, case_ids = process.experiment_data_thoracic_agenttest(is_sampling)

        # self.actions = actions
        states = states_dl
        # self.next_states = next_state_dl
        # self.rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        case_ids = torch.tensor(case_ids).unsqueeze(1)
        # self.unique_caseids = np.unique(case_ids)
        ############################################################################################################
        # load the agent model
        if reload==True:
            self.model.load_model(self.model_name, self.scheme,self.dataset_name)
        # result = pd.DataFrame(columns=['state', 'actions', 'reward', 'next_state', 'bis', 'caseid', 'step'])
        result = {'state': [], 'actions': [], 'reward': [], 'next_state': [], 'bis': [], 'caseid': [], 'step': []}

        caseid_unique_list = torch.unique(case_ids)
        cased_id_list = list(map(lambda x: x.item(), case_ids))
        # 将tensor转换为int
        caseid_unique_list_first_state = list(map(lambda x: states[cased_id_list.index(x)], caseid_unique_list)) # 用于存储每个caseid的第一个state

        # 测试
        # caseid_unique_list = caseid_unique_list[0:50]

        # for i in tqdm(range(len(caseid_unique_list))):
        for i in range(len(caseid_unique_list)):
            # agent的state后面的两位action累加、原始数据的state后两位action也是累加的，env model输入的state后两位action不累加，env model输出的state没有action和state_demographic
            action_cum = np.array([0] * self.num_agents)
            all_steps = cased_id_list.count(caseid_unique_list[i])
            state = caseid_unique_list_first_state[i]
            agent_input = state
            state_demographic = state.squeeze()[:4].detach().tolist()
            state = state[:-self.num_agents]

            for j in range(1, all_steps):
                actions = torch.tensor([0.0005, 0.0006])
                action_cum = action_cum + np.array(actions.tolist())
                env_input = torch.cat((state, actions), 0)
                env_input = np.array(env_input).reshape(1, -1)

                output = self.env_model.predict(env_input)
                next_state, reward, bis  = self.reformat_env_model_output(output, state_demographic, env_input,actions[0],actions[1])

                result['state'].append(state)
                result['actions'].append(actions)
                result['reward'].append(reward)
                result['next_state'].append(next_state)
                result['bis'].append(bis)
                result['caseid'].append(caseid_unique_list[i])
                result['step'].append(j)

                state = next_state
                agent_input = torch.cat((state, torch.tensor(action_cum)), 0)

        result_df = pd.DataFrame(result)
        #
        result_df = self.data_type_convert(result_df)

        result_df.to_csv(self.test_root + f'test_df_{self.scheme}_{dataset_name}_fixaction_bisreward.csv', index=False)
        print('test result saved')

    ############################################################################################################
    # BIS奖励: 在40 - 60之间时使用指数函数，超出范围时使用平方惩罚。
    def reward_function(self, bis, action1, action2):
        # Convert the inputs to PyTorch tensors if they are not already
        if isinstance(bis, (pd.Series, np.ndarray)):
            bis = torch.tensor(bis.values, dtype=torch.float32)
        if isinstance(action1, (pd.Series, np.ndarray)):
            action1 = torch.tensor(action1.values, dtype=torch.float32)
        if isinstance(action2, (pd.Series, np.ndarray)):
            action2 = torch.tensor(action2.values, dtype=torch.float32)
        # BIS奖励50=0.1428 40-60=[0.1142, 0.1714]) BIS奖励: 越接近50越好
        bis_ideal_range = torch.tensor([0.1142, 0.1714])
        # Calculate the means and standard deviations
        bis_ideal_mean = bis_ideal_range.mean()
        bis_ideal_std = (bis_ideal_range[1] - bis_ideal_range[0])

        bis_reward = torch.exp(-((bis - bis_ideal_mean) ** 2) / (2 * (bis_ideal_std ** 2)))

        # bis_reward=1.156 * (np.exp(-((bis - 0.1428) ** 2) / 0.00040898) - 0.1353)
        ############################################################################################################
        # # # Action1+Action2的总和越小越好
        # total = action1 + action2
        # # Exponential decay
        # # reward = torch.exp(-total)
        # # Inverse relationship
        # reward = 1 / (1 + total)  # Adding 1 to avoid division by zero
        # # # # 总奖励
        # total_reward = bis_reward + reward
        return bis_reward

############################################################################################################
    def bis_reward_function(self,bis):
        # Convert the inputs to PyTorch tensors if they are not already
        if isinstance(bis, (pd.Series, np.ndarray)):
            bis = torch.tensor(bis.values, dtype=torch.float32)

        bis_ideal_range = torch.tensor([0.1142, 0.1714])
        # Calculate the means and standard deviations
        bis_ideal_mean = bis_ideal_range.mean()
        bis_ideal_std = (bis_ideal_range[1] - bis_ideal_range[0])

        bis_reward = torch.exp(-((bis - bis_ideal_mean) ** 2) / (2 * (bis_ideal_std ** 2)))

        # Combine the rewards
        total_reward = bis_reward

        return total_reward
    ############################################################################################################
    def reformat_env_model_output(self, output, state_demographic, env_input, action1, action2):
        output_sub = output.squeeze().tolist()
        # 拼接 state_values 和 output_values
        next_state_values = state_demographic + output_sub
        # 创建新的 next_state 张量
        next_state = torch.tensor([next_state_values])
        next_state = next_state.squeeze()
        # 生成奖励
        bis = env_input.squeeze()[4]
        # bis_re = next_state.squeeze()[4].tolist()
        bis_re = next_state.squeeze()[4]
        reward = self.reward_function(bis_re,action1,action2).tolist()
        # reward =self.bis_reward_function(bis_re).tolist()
        return next_state, reward, bis



















