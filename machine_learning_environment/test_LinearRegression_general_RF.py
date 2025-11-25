from common.data_processing_test import proceesing
from common.read_yaml import ReadYaml as read_yaml
import pandas as pd
import numpy as np
#打印metrics
from sklearn.metrics import mean_squared_error, r2_score
#保存模型
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

param = read_yaml()
param = param.read()
address_param = param['basic_config']
data_root = address_param['data_root']
predata_root = address_param['predata_root']
new_weight_root= address_param['weight_root']
image_root = address_param['image_root']

device = "cuda"
process = proceesing()

# 加载数据

trajectory_test_general = pd.read_csv(predata_root + 'trajectory_test_general4.csv')
#############################
def data_process(data):
    X_1 = data['state_dl']
    X_1 = X_1.str.split(',', expand=True)
    X_1.columns = ['age', 'sex', 'weight', 'height','BIS/BIS','Orchestra/PPF20_CE', 'Orchestra/PPF20_CP','Orchestra/RFTN20_CE','Orchestra/RFTN20_CP',
                                               'Solar8000/HR','Solar8000/ART_MBP','Solar8000/VENT_RR', 'Solar8000/BT']
    X_2 = data[['Orchestra/PPF20_VOL_single', 'Orchestra/RFTN20_VOL_single']]
    X = pd.concat([X_1, X_2], axis=1)
    y = data['next_state_dl_env']
    y = y.str.split(',', expand=True)
    y.columns = ['BIS/BIS', 'Orchestra/PPF20_CE', 'Orchestra/PPF20_CP', 'Orchestra/RFTN20_CE',
                'Orchestra/RFTN20_CP', 'Solar8000/HR', 'Solar8000/ART_MBP', 'Solar8000/VENT_RR', 'Solar8000/BT']
    return X, y

#####################################################加载模型########################################################
# 加载模型
rf_model = joblib.load(new_weight_root + 'rf_general_2.pkl')

#####################################################加载模型########################################################
# 初始化一个空的列表来存储每个case的MSE
mse_scores_RF = []
mse_scores_GBR = []
mse_scores_XGB = []
mse_scores_SVR = []
r2_scores_RF = []
r2_scores_GBR = []
r2_scores_XGB = []
r2_scores_SVR = []
f1_scores_RF = []
precision_scores_RF = []
recall_scores_RF = []
# 初始化字典来存储每个模型的MSE、RMSE和R^2分数
metrics_dict = {
    'RF': {'MSE': [], 'RMSE': [], 'R2': []},
    'GBR': {'MSE': [], 'RMSE': [], 'R2': []},
    'XGB': {'MSE': [], 'RMSE': [], 'R2': []},
    'SVR': {'MSE': [], 'RMSE': [], 'R2': []}
}
# column_names 列表包含所有预期的列名，应当与 case_y_test 的列数匹配
column_names = ['BIS/BIS', 'Orchestra/PPF20_CE', 'Orchestra/PPF20_CP', 'Orchestra/RFTN20_CE',
                'Orchestra/RFTN20_CP', 'Solar8000/HR', 'Solar8000/ART_MBP', 'Solar8000/VENT_RR', 'Solar8000/BT']

# 分组交叉验证
predictions = []
########################################################################################
for caseid, case_data in trajectory_test_general.groupby('caseid'):
    # 提取当前case的测试数据
    case_X_test, case_y_test = data_process(case_data)
    # 进行预测
    case_y_pred_RF = rf_model.predict(case_X_test)


    ############################################
    # # 计算当前case的MSE和 r2_score
    case_mse_RF = mean_squared_error(case_y_test, case_y_pred_RF)
    mse_scores_RF.append(case_mse_RF)

    case_r2score_RF=r2_score(case_y_test, case_y_pred_RF)
    r2_scores_RF.append(case_r2score_RF)

    # 计算当前case的MSE和 r2_score,f1_score,precision_score,recall_score
    # case_f1score_RF = f1_score(case_y_test, case_y_pred_RF, average='weighted')
    # f1_scores_RF.append(case_f1score_RF)
    #
    # case_precisionscore_RF = precision_score(case_y_test, case_y_pred_RF, average='weighted')
    # precision_scores_RF.append(case_precisionscore_RF)
    #
    # case_recallscore_RF = recall_score(case_y_test, case_y_pred_RF, average='weighted')
    # recall_scores_RF.append(case_recallscore_RF)

    ############################################
    # 确保case_y_test是一个numpy数组
    case_y_test = np.array(case_y_test)
    # 对每个特征计算MSE, RMSE, 和 R^2 分数
    for i, column_name in enumerate(column_names):
        for model_name, case_y_pred in zip(['RF'],
                                           [case_y_pred_RF]):
            mse = mean_squared_error(case_y_test[:, i], case_y_pred[:, i])
            r2 = r2_score(case_y_test[:, i], case_y_pred[:, i])


            metrics_dict[model_name]['MSE'].append(mse)
            metrics_dict[model_name]['RMSE'].append(np.sqrt(mse))
            metrics_dict[model_name]['R2'].append(r2)
    ############################################
    # 创建一个新的DataFrame用于存储预测值
    case_y_pred_df_RF = pd.DataFrame(case_y_pred_RF, columns=column_names)

    case_y_test = pd.DataFrame(case_y_test, columns=column_names)
    # 给真实值和预测值的DataFrame加上标记
    case_y_test['model'] = 'GT'
    case_y_pred_df_RF['model'] = 'RF'

    # Use the index directly for timestep
    case_y_test.reset_index(drop=True, inplace=True)
    case_y_test['timestep'] = case_y_test.index
    case_y_pred_df_RF['timestep'] = case_y_pred_df_RF.index

    # 合并真实值和预测值的DataFrame
    combined_df = pd.concat([case_y_test, case_y_pred_df_RF])

    #添加caseid列
    combined_df['caseid'] = caseid
    # 重新设置索引，去除多级索引，因为我们已经有了Type列来区分数据类型
    combined_df.reset_index(drop=True, inplace=True)
    predictions.append(combined_df)  # This line was missing

# 将所有预测结果合并到一个DataFrame中
predictions_df = pd.concat(predictions)
#保存数据
# predictions_df.to_csv(predata_root + 'predictions_df_machinelr.csv', index=False)
########################################################################################
# 计算每种model的平均MSE
average_mse_RF = np.mean(mse_scores_RF)
average_mse_GBR = np.mean(mse_scores_GBR)
average_mse_XGB = np.mean(mse_scores_XGB)
average_mse_SVR = np.mean(mse_scores_SVR)
print(f"Average MSE_rf_model: {average_mse_RF}")
print(f"Average MSE_gbr_model: {average_mse_GBR}")
print(f"Average MSE_xgboost_model: {average_mse_XGB}")
print(f"Average MSE_svr: {average_mse_SVR}")
# 计算RMSE
average_rmse_RF = np.sqrt(average_mse_RF)
average_rmse_GBR = np.sqrt(average_mse_GBR)
average_rmse_XGB = np.sqrt(average_mse_XGB)
average_rmse_SVR = np.sqrt(average_mse_SVR)
print(f"Average RMSE_rf_model: {average_rmse_RF}")
print(f"Average RMSE_gbr_model: {average_rmse_GBR}")
print(f"Average RMSE_xgboost_model: {average_rmse_XGB}")
print(f"Average RMSE_svr: {average_rmse_SVR}")
# 计算每种model的平均R^2
average_r2_RF = np.mean(r2_scores_RF)
average_r2_GBR = np.mean(r2_scores_GBR)
average_r2_XGB = np.mean(r2_scores_XGB)
average_r2_SVR = np.mean(r2_scores_SVR)
print(f"Average R^2_rf_model: {average_r2_RF}")
print(f"Average R^2_gbr_model: {average_r2_GBR}")
print(f"Average R^2_xgboost_model: {average_r2_XGB}")
print(f"Average R^2_svr: {average_r2_SVR}")
# 计算每种model的平均f1_score
average_f1_RF = np.mean(f1_scores_RF)
print(f"Average f1_score_rf_model: {average_f1_RF}")
# 计算每种model的平均precision_score
average_precision_RF = np.mean(precision_scores_RF)
print(f"Average precision_score_rf_model: {average_precision_RF}")
# 计算每种model的平均recall_score
average_recall_RF = np.mean(recall_scores_RF)
print(f"Average recall_score_rf_model: {average_recall_RF}")

########################################################################################
# 将metrics_dict转换为DataFrame
metrics_df_list = []
for model, metrics in metrics_dict.items():
    for metric_name, values in metrics.items():
        for i, value in enumerate(values):
            metrics_df_list.append({
                'Model': model,
                'Feature': column_names[i % len(column_names)],
                'Metric': metric_name,
                'Value': value
            })

metrics_df = pd.DataFrame(metrics_df_list)

# 可以选择按模型、特征和指标分组来计算平均值
average_metrics_df = metrics_df.groupby(['Model', 'Feature', 'Metric']).mean().reset_index()
#将'Value'保留小数点后四位
average_metrics_df['Value'] = average_metrics_df['Value'].round(4)
# 保存性能指标
average_metrics_df.to_csv(predata_root + 'average_metrics_df_machinelr.csv', index=False)

# #去掉其中一个model的数据
# average_metrics_df = average_metrics_df[average_metrics_df['Model'] != 'SVR']

print(average_metrics_df)
########################################################################################

# ########################################################################################
########################################################################################
#在一张图上同时画出['model'] == 'RF'和['model'] == 'ground_truth']的所有feature的折线图，并且折线图具有置信区间的形式，并且这个折线图有两个y轴，右边的y轴显示的特征包括VENT_RR，BT，左边y轴是除了VENT_RR和BT之外的所有特征

###########################################################查看每个caseid的每个model的数据情况(ground truth)############################################################################
predictions_df.columns = ['BIS', 'PPF_CE', 'PPF_CP', 'RFTN_CE',
                'RFTN_CP', 'HR', 'MBP', 'RR', 'BT','model',
         'timestep', 'caseid']

# Assuming predictions_df is a previously defined DataFrame.
all_results_df = predictions_df[(predictions_df['model'] == 'GT') | (predictions_df['model'] == 'RF')]

# columns_to_convert = ['BIS', 'PPF20_CE', 'PPF20_CP', 'RFTN20_CE', 'RFTN20_CP', 'HR', 'ART_MBP', 'VENT_RR', 'BT']
columns_to_convert = ['BIS', 'PPF_CE', 'PPF_CP', 'RFTN_CE','RFTN_CP', 'HR', 'MBP', 'RR', 'BT']
for column in columns_to_convert:
    all_results_df[column] = pd.to_numeric(all_results_df[column])

################################################################################################################################################
# #############################################反归一化反归一化反归一化###########################################
def de_normalize(data):
    attributes_to_denormalize = ['BIS', 'PPF_CE', 'PPF_CP', 'RFTN_CE', 'RFTN_CP', 'HR', 'MBP', 'RR',
       'BT']
    for attr in attributes_to_denormalize:
        data[attr] = pd.to_numeric(data[attr])
        data[attr] = process.min_max_denormalize(data[attr])
    return data

all_results_df=de_normalize(all_results_df)

# #############################################反归一化反归一化反归一化###########################################
# 获取模型的唯一值，并逆序排列
unique_models = list(reversed(all_results_df['model'].unique()))
err_kws = {'alpha': 0.09}
# 创建图和y轴figsize=(15, 8)
fig, ax = plt.subplots()
# 为每个模型和指标选择一个颜色
palette = sns.color_palette("bright", len(unique_models) * len(['RR', 'BT']))
palette.reverse()
# 用于给线条命名的计数器
color_count = 0

for metric in ['RR', 'BT']:
    for model in unique_models:
        # 画出每个模型的指标线条
        sns.lineplot(
            data=all_results_df[all_results_df['model'] == model],
            x='timestep',
            y=metric,
            label=f'{metric}_{model}',
            color=palette[color_count],
            estimator='mean',
            errorbar='sd',
            err_kws=err_kws,
            ax=ax
        )
        color_count += 1

# 设置左侧y轴标题
ax.set_ylabel('RR/BT', fontsize=15)
# 设置x轴标题
ax.set_xlabel('Timestep(30sec)', fontsize=15)
# 调整坐标轴刻度大小
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
# 处理图例，放在图内并且不遮挡数据
ax.legend(loc='best', fontsize=11)
#将x轴和y轴的label距离图像调远一点
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
# 调整图像的位置以居中
plt.tight_layout()
# 保存图片
plt.savefig(image_root + 'ml_rr_bt_general.pdf',  bbox_inches='tight')
plt.show()

################################################################################################################################################
# 创建图和y轴figsize=(15, 8)
fig, ax = plt.subplots()
# 为每个模型和指标选择一个颜色
palette = sns.color_palette("bright", len(unique_models) * len(['PPF_CP','RFTN_CP']))
palette.reverse()
# 用于给线条命名的计数器
color_count = 0

for metric in ['PPF_CP','RFTN_CP']:
    for model in unique_models:
        # 画出每个模型的指标线条
        sns.lineplot(
            data=all_results_df[all_results_df['model'] == model],
            x='timestep',
            y=metric,
            label=f'{metric}_{model}',
            color=palette[color_count],
            estimator='mean',
            errorbar='sd',
            err_kws=err_kws,
            ax=ax
        )
        color_count += 1

# 设置左侧y轴标题
ax.set_ylabel('PPF_CP/RFTN_CP', fontsize=15)
# 设置x轴标题
ax.set_xlabel('Timestep(30sec)', fontsize=15)
# 设置y轴显示科学计数法
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# 调整坐标轴刻度大小
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
# 处理图例，放在图内并且不遮挡数据
ax.legend(loc='best', fontsize=11)
#将x轴和y轴的label距离图像调远一点
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
# 调整图像的位置以居中
plt.tight_layout()
# 保存图片
plt.savefig(image_root + 'ml_cp_general.pdf',  bbox_inches='tight')
plt.show()

################################################################################################################################################
# 创建图和y轴figsize=(15, 8)
fig, ax = plt.subplots()
# 为每个模型和指标选择一个颜色
palette = sns.color_palette("bright",len(unique_models) * len(['BIS','HR', 'MBP'])+1)
del palette[5]
palette.reverse()
# 用于给线条命名的计数器
color_count = 0

for metric in ['HR','BIS', 'MBP']:
    for model in unique_models:
        # 画出每个模型的指标线条
        sns.lineplot(
            data=all_results_df[all_results_df['model'] == model],
            x='timestep',
            y=metric,
            label=f'{metric}_{model}',
            color=palette[color_count],
            estimator='mean',
            errorbar='sd',
            err_kws=err_kws,
            ax=ax
        )
        color_count += 1

# 设置左侧y轴标题
ax.set_ylabel('BIS/HR/MBP', fontsize=15)
# 设置x轴标题
ax.set_xlabel('Timestep(30sec)', fontsize=15)
# 调整坐标轴刻度大小
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
# 处理图例，放在图内并且不遮挡数据
ax.legend(loc='best', fontsize=11)
#将x轴和y轴的label距离图像调远一点
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
# 调整图像的位置以居中
plt.tight_layout()
#保存图片
plt.savefig(image_root + 'ml_bis_hr_mbp_general.pdf',  bbox_inches='tight')
plt.show()

################################################################################################################################################
# 创建图和y轴figsize=(15, 8)
fig, ax = plt.subplots()
# 为每个模型和指标选择一个颜色
palette = sns.color_palette("bright",len(unique_models) * len(['PPF_CE','RFTN_CE']))
palette.reverse()
# 用于给线条命名的计数器
color_count = 0

for metric in ['PPF_CE','RFTN_CE']:
    for model in unique_models:
        # 画出每个模型的指标线条
        sns.lineplot(
            data=all_results_df[all_results_df['model'] == model],
            x='timestep',
            y=metric,
            label=f'{metric}_{model}',
            color=palette[color_count],
            estimator='mean',
            errorbar='sd',
            err_kws=err_kws,
            ax=ax
        )
        color_count += 1

# 设置左侧y轴标题
ax.set_ylabel('PPF_CE/RFTN_CE', fontsize=15)
# 设置x轴标题
ax.set_xlabel('Timestep(30sec)', fontsize=15)
#设置y轴显示科学计数法
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# 调整坐标轴刻度大小
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
# 处理图例，放在图内并且不遮挡数据
ax.legend(loc='lower center', fontsize=11)
#将x轴和y轴的label距离图像调远一点
ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
# 调整图像的位置以居中
plt.tight_layout()
#保存图片
plt.savefig(image_root + 'ml_ce_general.pdf',  bbox_inches='tight')
plt.show()

################################################################################################################################################

# joblib.dump(svr_pipeline, new_weight_root + 'svr_pipeline.pkl')
# print('svr_pipeline saved')
#
# #####################################################支持向量回归（SVR）########################################################

