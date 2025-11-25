import ast
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from common.read_yaml import ReadYaml as read_yaml

import random


# address
param = read_yaml()
param = param.read()

class proceesing:

    def __init__(self):
        basic_param = param['basic_config']
        self.predata_root = basic_param['predata_root']
        self.data_root = basic_param['data_root']


##############################################test setting########################################################

    def experiment_data_general(self, train_test='trajectory_train'):
        # 读取数据
        self.normalized_data = pd.read_csv(self.predata_root + 'normalized_data_model_add_state_pd_differ.csv')
        #判断是否有空值
        # print(self.normalized_data.isnull().sum())
        # self.normalized_data = self.normalized_data.groupby('caseid').head(900)
        self.normalized_data_model = self.normalized_data[(self.normalized_data['department'] == 'General surgery')]
        # 创建空DataFrame
        self.new_df = pd.DataFrame()
        # 获取输入和输出数据  # 单个值
        for patient_id, group in self.normalized_data_model.groupby('caseid'):
            patient_data = self.normalized_data_model.loc[self.normalized_data_model['caseid'] == patient_id]
            # patient_data = self.merge_all_data_final.loc[self.merge_all_data_final['caseid'] == patient_id].iloc[6:]
            # state, action, reward, next_state
            # patient_data['state_dl'] = patient_data[['BIS/BIS']]
            patient_data['next_state_dl_env'] = patient_data['next_dl_state_dl'].shift(-1)
            patient_data['next_state_dl'] = patient_data['state_dl'].shift(-1)

            ######################################################################################################
            # patient_data['reward'] = self.bis_reward_function(patient_data['BIS/BIS'])
            ######################################################################################################
            patient_data['reward'] = self.reward_function(patient_data['BIS/BIS'],
                                                          patient_data['Orchestra/PPF20_VOL_single'],
                                                          patient_data['Orchestra/RFTN20_VOL_single'])
            # ######################################################################################################

            patient_data['reward'] = patient_data['reward'].shift(-1)
            ######################################################################################################
            patient_data['action_volume_agent'] = patient_data['action_volume'].shift(1).fillna('0,0')
            ######################################################################################################
            # 删除最后一行，因为最后一行没有下一行
            patient_data = patient_data.drop(patient_data.index[-1])

            self.new_df = pd.concat([self.new_df, patient_data], axis=0)

        # 将数据按照病人分组，获取病人ID
        case_ids = self.new_df['caseid'].unique()

        # 随机选择80%的病人作为训练集,并且保证测试集中每个caseid的时间步的长度不能大于1000
        train_case_ids, test_case_ids = train_test_split(case_ids, test_size=0.2)

        # 根据训练集和测试集的病人ID筛选数据
        self.train_data = self.new_df[self.new_df['caseid'].isin(train_case_ids)]
        self.test_data = self.new_df[self.new_df['caseid'].isin(test_case_ids)]

        # 输出训练集和测试集的形状
        print('Training data shape:', self.train_data.shape)
        print('Testing data shape:', self.test_data.shape)
       # print('Valid data shape:', self.valid_data.shape)

        if train_test == 'trajectory_train':
            # return self.train_data#,self.valid_data
            return self.train_data
        if train_test == 'trajectory_test_final':
            return self.test_data

    def experiment_data_Thoracic(self, train_test='trajectory_train'):
        # 读取数据
        self.normalized_data = pd.read_csv(self.predata_root + 'normalized_data_model_add_state_pd_differ.csv')
        # self.normalized_data = self.normalized_data.groupby('caseid').head(900)
        self.normalized_data_model = self.normalized_data[(self.normalized_data['department'] == 'Thoracic surgery')]
        # 创建空DataFrame
        self.new_df = pd.DataFrame()

        for patient_id, group in self.normalized_data_model.groupby('caseid'):
            patient_data = self.normalized_data_model.loc[self.normalized_data_model['caseid'] == patient_id]
            # patient_data = self.merge_all_data_final.loc[self.merge_all_data_final['caseid'] == patient_id].iloc[6:]
            ######################################################################################################
            # patient_data['reward'] = self.bis_reward_function(patient_data['BIS/BIS'])
            ######################################################################################################
            patient_data['reward'] = self.reward_function(patient_data['BIS/BIS'],
                                                          patient_data['Orchestra/PPF20_VOL_single'],
                                                          patient_data['Orchestra/RFTN20_VOL_single'])
            ######################################################################################################

            # patient_data['state_dl'] = patient_data[['BIS/BIS']] ,patient_data['Solar8000/HR'],patient_data['Solar8000/ART_MBP']
            patient_data['next_state_dl_env'] = patient_data['next_dl_state_dl'].shift(-1)
            patient_data['next_state_dl'] = patient_data['state_dl'].shift(-1)

            # patient_data['next_state_dl'] = patient_data['state_dl'].shift(-1)
            # patient_data['next_state_dl_env'] = patient_data['next_dl_state_dl'].shift(-1)
            # patient_data['next_state_pkdl_env'] = patient_data['next_pkdl_state_dl'].shift(-1)
            # patient_data['next_state_pk_dl'] = patient_data['state_pk_dl'].shift(-1)
            patient_data['reward'] = patient_data['reward'].shift(-1)
            # patient_data['next_state_pk_pd_dl'] = patient_data['state_pk_pd_dl'].shift(-1)
            patient_data['action_volume_agent'] = patient_data['action_volume'].shift(1).fillna('0,0')
            # 删除最后一行，因为最后一行没有下一行
            patient_data = patient_data.drop(patient_data.index[-1])

            self.new_df = pd.concat([self.new_df, patient_data], axis=0)



        # 将数据按照病人分组，获取病人ID
        case_ids = self.new_df['caseid'].unique()
        # 随机选择80%的病人作为训练集
        train_case_ids, test_case_ids = train_test_split(case_ids, test_size=0.2)
        # 根据训练集和测试集的病人ID筛选数据
        self.train_data = self.new_df[self.new_df['caseid'].isin(train_case_ids)]
        self.test_data = self.new_df[self.new_df['caseid'].isin(test_case_ids)]

        # 输出训练集和测试集的形状
        print('Training data shape:', self.train_data.shape)
        print('Testing data shape:', self.test_data.shape)
        # print('Valid data shape:', self.valid_data.shape)

        if train_test == 'trajectory_train':
            # return self.train_data#,self.valid_data
            return self.train_data
        if train_test == 'trajectory_test_final':
            return self.test_data



    def experiment_data_general_agenttrain(self, sampling = False):
        # 读取数据
        self.trajectory_train_general = pd.read_csv(self.predata_root + 'trajectory_train_general4.csv')

        ##############################将age，action volume并到state中############################################
        self.trajectory_train_general['state_agent'] = self.trajectory_train_general[['state_dl', 'action_volume_agent']].astype(str).agg(','.join, axis=1)
        ##############################将age，action volume并到state中############################################
        ##############################将age，action volume并到state中############################################
        self.trajectory_train_general['state_agent'] = self.trajectory_train_general['state_agent'].apply(lambda x: str(x))
        if sampling:
            case_id_all = torch.tensor(self.trajectory_train_general['caseid'].values, dtype=torch.int32)
            case_id_chosen = random.sample(list(set(case_id_all)), 5)
            case_id_chosen = list(map(lambda x: float(x), case_id_chosen))
            self.trajectory_train_general = self.trajectory_train_general[self.trajectory_train_general['caseid'].isin(case_id_chosen)]
        states_dl = self.trajectory_train_general['state_agent'].apply(ast.literal_eval)
        states_dl = torch.tensor(states_dl.values.tolist())

        self.trajectory_train_general['action_single'] = self.trajectory_train_general['action_single'].apply(lambda x: str(x))
        actions = self.trajectory_train_general['action_single'].apply(ast.literal_eval)
        actions = torch.tensor(actions.values.tolist())

        self.trajectory_train_general['next_state_agent'] = self.trajectory_train_general[['next_state_dl', 'action_volume']].astype(str).agg(
            ','.join, axis=1)
        self.trajectory_train_general['next_state_agent'] = self.trajectory_train_general['next_state_agent'].apply(lambda x: str(x))
        next_state_dl = self.trajectory_train_general['next_state_agent'].apply(ast.literal_eval)
        next_state_dl = torch.tensor(next_state_dl.values.tolist())

        # self.train_data['next_state_pk_dl'] = self.train_data['next_state_pk_dl'].apply(lambda x: str(x))
        # next_state_pk_dl = self.train_data['next_state_pk_dl'].apply(ast.literal_eval)
        # next_state_pk_dl = torch.tensor(next_state_pk_dl.values.tolist())

        rewards = torch.tensor(self.trajectory_train_general['reward'].values, dtype=torch.float32)
        case_ids = torch.tensor(self.trajectory_train_general['caseid'].values, dtype=torch.int32)

        ######################################################################################################
        return states_dl,actions, next_state_dl,rewards, case_ids

    def experiment_data_thoracic_agenttrain(self, sampling = False):
        # 读取数据
        self.trajectory_train_thoracic = pd.read_csv(self.predata_root + 'trajectory_train_Thoracic4.csv')

        ##############################将age，action volume并到state中############################################
        self.trajectory_train_thoracic['state_agent'] = self.trajectory_train_thoracic[
            ['state_dl', 'action_volume_agent']].astype(str).agg(','.join, axis=1)
        ##############################将age，action volume并到state中############################################
        ##############################将age，action volume并到state中############################################
        self.trajectory_train_thoracic['state_agent'] = self.trajectory_train_thoracic['state_agent'].apply(
            lambda x: str(x))

        if sampling:
            case_id_all = torch.tensor(self.trajectory_train_thoracic['caseid'].values, dtype=torch.int32)
            case_id_chosen = random.sample(list(set(case_id_all)), 5)
            case_id_chosen = list(map(lambda x: float(x), case_id_chosen))
            self.trajectory_train_thoracic = self.trajectory_train_thoracic[self.trajectory_train_thoracic['caseid'].isin(case_id_chosen)]

        states_dl = self.trajectory_train_thoracic['state_agent'].apply(ast.literal_eval)
        states_dl = torch.tensor(states_dl.values.tolist())

        self.trajectory_train_thoracic['action_single'] = self.trajectory_train_thoracic['action_single'].apply(
            lambda x: str(x))
        actions = self.trajectory_train_thoracic['action_single'].apply(ast.literal_eval)
        actions = torch.tensor(actions.values.tolist())

        self.trajectory_train_thoracic['next_state_agent'] = self.trajectory_train_thoracic[
            ['next_state_dl', 'action_volume']].astype(str).agg(
            ','.join, axis=1)
        self.trajectory_train_thoracic['next_state_agent'] = self.trajectory_train_thoracic['next_state_agent'].apply(
            lambda x: str(x))
        next_state_dl = self.trajectory_train_thoracic['next_state_agent'].apply(ast.literal_eval)
        next_state_dl = torch.tensor(next_state_dl.values.tolist())

        # self.train_data['next_state_pk_dl'] = self.train_data['next_state_pk_dl'].apply(lambda x: str(x))
        # next_state_pk_dl = self.train_data['next_state_pk_dl'].apply(ast.literal_eval)
        # next_state_pk_dl = torch.tensor(next_state_pk_dl.values.tolist())

        rewards = torch.tensor(self.trajectory_train_thoracic['reward'].values, dtype=torch.float32)
        case_ids = torch.tensor(self.trajectory_train_thoracic['caseid'].values, dtype=torch.int32)
        ######################################################################################################
        return states_dl, actions, next_state_dl, rewards, case_ids


    def experiment_data_general_agenttest(self, sampling = False):
        # 读取数据
        self.trajectory_train_general = pd.read_csv(self.predata_root + 'trajectory_test_general4.csv')

        ##############################将age，action volume并到state中############################################
        self.trajectory_train_general['state_agent'] = self.trajectory_train_general[['state_dl', 'action_volume_agent']].astype(str).agg(','.join, axis=1)
        ##############################将age，action volume并到state中############################################
        ##############################将age，action volume并到state中############################################
        self.trajectory_train_general['state_agent'] = self.trajectory_train_general['state_agent'].apply(lambda x: str(x))
        if sampling==True:
            case_id_all = torch.tensor(self.trajectory_train_general['caseid'].values, dtype=torch.int32)
            case_id_chosen = random.sample(list(set(case_id_all)), 5)
            case_id_chosen = list(map(lambda x: float(x), case_id_chosen))
            self.trajectory_train_general = self.trajectory_train_general[self.trajectory_train_general['caseid'].isin(case_id_chosen)]
        states_dl = self.trajectory_train_general['state_agent'].apply(ast.literal_eval)
        states_dl = torch.tensor(states_dl.values.tolist())

        self.trajectory_train_general['action_single'] = self.trajectory_train_general['action_single'].apply(lambda x: str(x))
        actions = self.trajectory_train_general['action_single'].apply(ast.literal_eval)
        actions = torch.tensor(actions.values.tolist())

        self.trajectory_train_general['next_state_agent'] = self.trajectory_train_general[['next_state_dl', 'action_volume']].astype(str).agg(
            ','.join, axis=1)
        self.trajectory_train_general['next_state_agent'] = self.trajectory_train_general['next_state_agent'].apply(lambda x: str(x))
        next_state_dl = self.trajectory_train_general['next_state_agent'].apply(ast.literal_eval)
        next_state_dl = torch.tensor(next_state_dl.values.tolist())

        # self.train_data['next_state_pk_dl'] = self.train_data['next_state_pk_dl'].apply(lambda x: str(x))
        # next_state_pk_dl = self.train_data['next_state_pk_dl'].apply(ast.literal_eval)
        # next_state_pk_dl = torch.tensor(next_state_pk_dl.values.tolist())

        rewards = torch.tensor(self.trajectory_train_general['reward'].values, dtype=torch.float32)
        case_ids = torch.tensor(self.trajectory_train_general['caseid'].values, dtype=torch.int32)

        ######################################################################################################
        return states_dl,actions, next_state_dl,rewards, case_ids

    def experiment_data_thoracic_agenttest(self, sampling = False):
        # 读取数据
        self.trajectory_train_thoracic = pd.read_csv(self.predata_root + 'trajectory_test_Thoracic4.csv')

        ##############################将age，action volume并到state中############################################
        self.trajectory_train_thoracic['state_agent'] = self.trajectory_train_thoracic[
            ['state_dl', 'action_volume_agent']].astype(str).agg(','.join, axis=1)
        ##############################将age，action volume并到state中############################################
        ##############################将age，action volume并到state中############################################
        self.trajectory_train_thoracic['state_agent'] = self.trajectory_train_thoracic['state_agent'].apply(
            lambda x: str(x))

        if sampling==True:
            case_id_all = torch.tensor(self.trajectory_train_thoracic['caseid'].values, dtype=torch.int32)
            case_id_chosen = random.sample(list(set(case_id_all)), 5)
            case_id_chosen = list(map(lambda x: float(x), case_id_chosen))
            self.trajectory_train_thoracic = self.trajectory_train_thoracic[self.trajectory_train_thoracic['caseid'].isin(case_id_chosen)]

        states_dl = self.trajectory_train_thoracic['state_agent'].apply(ast.literal_eval)
        states_dl = torch.tensor(states_dl.values.tolist())

        self.trajectory_train_thoracic['action_single'] = self.trajectory_train_thoracic['action_single'].apply(
            lambda x: str(x))
        actions = self.trajectory_train_thoracic['action_single'].apply(ast.literal_eval)
        actions = torch.tensor(actions.values.tolist())

        self.trajectory_train_thoracic['next_state_agent'] = self.trajectory_train_thoracic[
            ['next_state_dl', 'action_volume']].astype(str).agg(
            ','.join, axis=1)
        self.trajectory_train_thoracic['next_state_agent'] = self.trajectory_train_thoracic['next_state_agent'].apply(
            lambda x: str(x))
        next_state_dl = self.trajectory_train_thoracic['next_state_agent'].apply(ast.literal_eval)
        next_state_dl = torch.tensor(next_state_dl.values.tolist())

        # self.train_data['next_state_pk_dl'] = self.train_data['next_state_pk_dl'].apply(lambda x: str(x))
        # next_state_pk_dl = self.train_data['next_state_pk_dl'].apply(ast.literal_eval)
        # next_state_pk_dl = torch.tensor(next_state_pk_dl.values.tolist())

        rewards = torch.tensor(self.trajectory_train_thoracic['reward'].values, dtype=torch.float32)
        case_ids = torch.tensor(self.trajectory_train_thoracic['caseid'].values, dtype=torch.int32)
        ######################################################################################################
        return states_dl, actions, next_state_dl, rewards, case_ids

    def experiment_data_all(self, train_test='general'):
        # 读取数据
        self.normalized_data_all = pd.read_csv(self.predata_root + 'normalized_data_model_add_state_pd_differ.csv')
        self.normalized_data_thoracic = self.normalized_data_all[(self.normalized_data_all['department'] == 'Thoracic surgery')]
        self.normalized_data_general = self.normalized_data_all[(self.normalized_data_all['department'] == 'General surgery')]
        if train_test == 'thoracic':
            self.normalized_data_thoracic['action_single'] = self.normalized_data_thoracic['action_single'].apply(lambda x: str(x))
            actions_thoracic = self.normalized_data_thoracic['action_single'].apply(ast.literal_eval)
            actions_thoracic = torch.tensor(actions_thoracic.values.tolist())
            return actions_thoracic
        if train_test == 'general':
            self.normalized_data_general['action_single'] = self.normalized_data_general['action_single'].apply(lambda x: str(x))
            actions_general = self.normalized_data_general['action_single'].apply(ast.literal_eval)
            actions_general = torch.tensor(actions_general.values.tolist())
            return actions_general

        ######################################################################################################


    # 对数据进行归一化
    def get_normalize_data(self, data):
    #     normalize_data = data[[ 'Orchestra/PPF20_VOL', 'Orchestra/PPF20_CE', 'Orchestra/PPF20_CP', 'Orchestra/RFTN20_VOL', 'Orchestra/RFTN20_CE',
    # 'Orchestra/RFTN20_CP', 'Solar8000/HR', 'Solar8000/ART_MBP', 'BIS/BIS', 'Orchestra/PPF20_VOL_single', 'Orchestra/RFTN20_VOL_single', 'age']]
        normalize_data = data[
            ['Orchestra/PPF20_VOL', 'Orchestra/PPF20_CE', 'Orchestra/PPF20_CP', 'Orchestra/RFTN20_VOL',
             'Orchestra/RFTN20_CE',
             'Orchestra/RFTN20_CP', 'BIS/BIS', 'Orchestra/PPF20_VOL_single',
             'Orchestra/RFTN20_VOL_single', 'age', 'sex', 'weight', 'height',
                                 'Solar8000/HR', 'Solar8000/VENT_RR', 'Solar8000/BT','Solar8000/ART_MBP']]
        # 将数据框中的所有列转换为数字类型
        normalize_data = normalize_data.apply(pd.to_numeric, errors='coerce')

        self.max_val = normalize_data.max().max()  # 最大值
        self.min_val = normalize_data.min().min()  # 最小值
        normalize_data = normalize_data.apply(self.min_max_normalize)

        normalize_data['time'] = data['timeBIS/BIS']
        normalize_data['caseid'] = data['caseid']
        normalize_data['department'] = data['department']
        normalize_data['timestep'] = data['timestep']

        return normalize_data
    #
    def min_max_denormalize(self, data):
        # 反归一化
        min_val=0.0
        max_val=350.0
        denormalized_data = data * (max_val - min_val) + min_val
        return denormalized_data
        # if single_or_multi == 'multi':
        #     denormalized_data_list = [x * (max_val - min_val) + min_val for x in data]
        #     return denormalized_data_list
        # else:
        #     denormalized_data = data * (max_val - min_val) + min_val
        #     return denormalized_data


    def min_max_normalize(self, data, single_or_multi='multi'):
        # 归一化
        if single_or_multi == 'multi':
            normalized_data_list = [(x - self.min_val) / (self.max_val - self.min_val) for x in data]
            return normalized_data_list
        else:
            normalized_data_single = (data - self.min_val) / (self.max_val - self.min_val)
            return normalized_data_single
######################################################################################################
    def reward_function(self, bis, action1, action2):
        # Convert the inputs to PyTorch tensors if they are not already
        if isinstance(bis, (pd.Series, np.ndarray)):
            bis = torch.tensor(bis.values, dtype=torch.float32)
        if isinstance(action1, (pd.Series, np.ndarray)):
            action1 = torch.tensor(action1.values, dtype=torch.float32)
        if isinstance(action2, (pd.Series, np.ndarray)):
            action2 = torch.tensor(action2.values, dtype=torch.float32)
        # BIS奖励50=0.1428 40-60=[0.1142, 0.1714])
        # BIS奖励
        # BIS奖励: 越接近50越好
        ############################################################################################################
        # # bis_reward = torch.exp(-torch.abs(bis - 0.1428))  # 50 对应的归一化值为 0.1428
        bis_ideal_range = torch.tensor([0.1142, 0.1714])
        # Calculate the means and standard deviations
        bis_ideal_mean = bis_ideal_range.mean()
        bis_ideal_std = (bis_ideal_range[1] - bis_ideal_range[0])

        bis_reward = torch.exp(-((bis - bis_ideal_mean) ** 2) / (2 * (bis_ideal_std ** 2)))
        ############################################################################################################
        # bis_reward = 1.156 * (np.exp(-((bis - 0.1428) ** 2) / 0.00040898) - 0.1353)
        # # Action1奖励: 值越接近0越好
        ############################################################################################################
        # # Action1+Action2的总和越小越好
        # total = action1 + action2
        # # Exponential decay
        # # reward = torch.exp(-total)
        # # Inverse relationship
        # reward = 1 / (1 + total)  # Adding 1 to avoid division by zero
        # # # # 总奖励
        # total_reward = bis_reward + reward
        return bis_reward
######################################################################################################
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
    # def reward_function(self,bis):
    #     # Convert the inputs to PyTorch tensors if they are not already
    #     if isinstance(bis, (pd.Series, np.ndarray)):
    #         bis = torch.tensor(bis.values, dtype=torch.float32)
    #     # if isinstance(vent_rr, (pd.Series, np.ndarray)):
    #     #     vent_rr = torch.tensor(vent_rr.values, dtype=torch.float32)
    #     # if isinstance(art_mbp, (pd.Series, np.ndarray)):
    #     #     art_mbp = torch.tensor(art_mbp.values, dtype=torch.float32)
    #
    #     # Define ideal ranges as tensors 40 <= bis <= 60   60<=hr<=100   70 <= map <= 110
    #     bis_ideal_range = torch.tensor([0.1142, 0.1714])
    #     # vent_rr_ideal_range = torch.tensor([0.1714, 0.2857])
    #     # art_mbp_ideal_range = torch.tensor([0.2, 0.3142])
    #
    #     # Calculate the means and standard deviations
    #     bis_ideal_mean = bis_ideal_range.mean()
    #     bis_ideal_std = (bis_ideal_range[1] - bis_ideal_range[0])
    #
    #     # vent_rr_ideal_mean = vent_rr_ideal_range.mean()
    #     # vent_rr_ideal_std = (vent_rr_ideal_range[1] - vent_rr_ideal_range[0])
    #     #
    #     # art_mbp_ideal_mean = art_mbp_ideal_range.mean()
    #     # art_mbp_ideal_std = (art_mbp_ideal_range[1] - art_mbp_ideal_range[0])
    #
    #     # Calculate the rewards
    #     # bis_reward = 1 - torch.abs((bis - bis_ideal_mean) / bis_ideal_std)
    #     # vent_rr_reward = 1 - torch.abs((vent_rr - vent_rr_ideal_mean) / vent_rr_ideal_std)
    #     # art_mbp_reward = 1 - torch.abs((art_mbp - art_mbp_ideal_mean) / art_mbp_ideal_std)
    #
    #
    #     bis_reward = torch.exp(-((bis - bis_ideal_mean) ** 2) / (2 * (bis_ideal_std ** 2)))
    #     # vent_rr_reward = torch.exp(-((vent_rr - vent_rr_ideal_mean) ** 2) / (2 * (vent_rr_ideal_std ** 2)))
    #     # art_mbp_reward = torch.exp(-((art_mbp - art_mbp_ideal_mean) ** 2) / (2 * (art_mbp_ideal_std ** 2)))
    #
    #     # Combine the rewards
    #     total_reward = bis_reward
    #
    #     return total_reward

        ######################################################################################################
    # BIS奖励: 在40 - 60之间时使用指数函数，超出范围时使用平方惩罚。
    # Action1和Action2:在0 - 14之间时，使用指数函数，值越小奖励越高。超出范围时，使用平方惩罚。

    ######################################################################################################
    ################################################原reward function######################################################

    #
    # def reward_function(self,bis, hr, map, single_or_multi='single'):
    #     if single_or_multi == 'single':
    #         # 初始化奖励
    #         reward = 0
    #         # 对BIS值进行奖励/惩罚
    #         if 40 <= bis <= 60:
    #             reward += 1  # BIS在目标范围内，给予正奖励
    #         else:
    #             reward -= abs(bis - 50) / 10  # BIS不在目标范围内，根据偏离程度给予惩罚
    #         # 对HR值进行奖励/惩罚
    #         if 60 <= hr <= 100:
    #             reward += 1  # HR在目标范围内，给予正奖励
    #         else:
    #             reward -= abs(hr - 80) / 20  # HR不在目标范围内，根据偏离程度给予惩罚
    #         # 对MAP值进行奖励/惩罚
    #         if 65 <= map <= 75:
    #             reward += 1  # MAP在目标范围内，给予正奖励
    #         else:
    #             reward -= abs(map - 70) / 5  # MAP不在目标范围内，根据偏离程度给予惩罚
    #         return reward
    #     else:
    #         # 初始化奖励数组
    #         reward_array = np.zeros(bis.shape)
    #
    #         # 对BIS值进行奖励/惩罚
    #         reward_array += np.where((bis >= 40) & (bis <= 60), 1, -np.abs(bis - 50) / 10)
    #
    #         # 对HR值进行奖励/惩罚
    #         reward_array += np.where((hr>= 60) & (hr <= 100), 1, -np.abs(hr - 80) / 20)
    #         # 对MAP值进行奖励/惩罚
    #         reward_array += np.where((map>= 65) & (map <= 75), 1, -np.abs(map - 70) / 5)
    #         return reward_array
    # # def reward_function(self, bis, single_or_multi='single'):
    #     # 0.09654075178214226
    #     if single_or_multi == 'single':
    #         return math.exp(- 0.08 *abs(bis - 0.09654))
    #
    #     else:
    #         return list(map(lambda x: math.exp(- 0.08 * abs(x - 0.09654)), bis))




 #
    # def reward_function(self, bis_series,time_series, total_time, single_or_multi='multi'):
    #     # 定义BIS目标范围 40 <= bis <= 60
    #     BIS_TARGET_LOW = 0.1142
    #     BIS_TARGET_HIGH = 0.1714
    #     BIS_OPTIMAL = 0.1428
    #
    #     if np.isscalar(bis_series):
    #         bis_series = pd.Series([bis_series])
    #
    #     # 定义时间相关系数的 Series
    #     TIME_DECAY_FACTOR = 1 - (time_series / total_time)
    #
    #     # 创建一个与 bis_series 等长的初始奖励 Series，并设置默认奖励值为 -1
    #     reward_series = pd.Series(-1, index=bis_series.index)
    #     # 如果BIS在目标范围内，给予正奖励
    #     reward_series[(bis_series >= BIS_TARGET_LOW) & (bis_series <= BIS_TARGET_HIGH)] = 1
    #     # 根据BIS距离目标值的距离调整奖励
    #     reward_series *= (1 - abs(bis_series - BIS_OPTIMAL) / BIS_OPTIMAL)
    #     # 在时间序列即将结束时，减小action的影响力，增加BIS的值
    #     # # 假设最后20%的时间准备结束手术
    #     # if single_or_multi == 'multi':
    #     #     reward_series[time_series > total_time * 0.9] *= TIME_DECAY_FACTOR
    #     # else:
    #     #     if (time_series > total_time * 0.9):
    #     #         reward_series *= TIME_DECAY_FACTOR
    #
    #     return reward_series
        #考虑MBP和HR的奖励和惩罚
        # #定义MBP和HR的目标范围 60<=hr<=100   70 <= map <= 110
        # if isinstance(vent_hr, (pd.Series, np.ndarray)):
        #     vent_hr = torch.tensor(vent_hr.values, dtype=torch.float32)
        # if isinstance(art_mbp, (pd.Series, np.ndarray)):
        #     art_mbp = torch.tensor(art_mbp.values, dtype=torch.float32)
        #
        # vent_hr_ideal_range = torch.tensor([0.1714, 0.2857])
        # art_mbp_ideal_range = torch.tensor([0.2, 0.3142])
        #
        # # Calculate the means and standard deviations
        # vent_hr_ideal_mean = vent_hr_ideal_range.mean()
        # vent_hr_ideal_std = (vent_hr_ideal_range[1] - vent_hr_ideal_range[0])
        # art_mbp_ideal_mean = art_mbp_ideal_range.mean()
        # art_mbp_ideal_std = (art_mbp_ideal_range[1] - art_mbp_ideal_range[0])
        #
        # # Calculate the rewards
        # vent_hr_reward = 1 - torch.abs((vent_hr - vent_hr_ideal_mean) / vent_hr_ideal_std)
        # art_mbp_reward = 1 - torch.abs((art_mbp - art_mbp_ideal_mean) / art_mbp_ideal_std)
        #
        # vent_hr_reward_numpy = vent_hr_reward.numpy()
        # art_mbp_reward_numpy = art_mbp_reward.numpy()
        #
        # # Convert the numpy arrays to Pandas Series
        # vent_hr_reward_series = pd.Series(vent_hr_reward_numpy)
        # art_mbp_reward_series = pd.Series(art_mbp_reward_numpy)
        # #总奖励值是三个指标的奖励值的和
        # reward_series = reward_series + vent_hr_reward_series + art_mbp_reward_series
