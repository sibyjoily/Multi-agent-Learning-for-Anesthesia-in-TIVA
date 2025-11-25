from common.data_processing_test import proceesing
from common.read_yaml import ReadYaml as read_yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#打印metrics
from sklearn.metrics import mean_squared_error, r2_score
#保存模型
import joblib

param = read_yaml()
param = param.read()
address_param = param['address']
common_param = param['common']
data_root = address_param['data_root']
predata_root = address_param['predata_root']
new_weight_root= address_param['new_weight_root']
image_root = address_param['image_root']

device = "cuda"
process = proceesing()

# 加载数据
trajectory_train_general = pd.read_csv(predata_root + 'trajectory_train_general2.csv')
trajectory_test_general = pd.read_csv(predata_root + 'trajectory_test_general2.csv')
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
    y.columns =  ['BIS/BIS', 'Orchestra/PPF20_CE', 'Orchestra/PPF20_CP', 'Orchestra/RFTN20_CE',
                'Orchestra/RFTN20_CP', 'Solar8000/HR', 'Solar8000/ART_MBP', 'Solar8000/VENT_RR', 'Solar8000/BT']
    return X, y
# 将所有数据用于训练模型
X_train, y_train = data_process(trajectory_train_general)
#####################################################加载模型########################################################
# 加载模型
rf_model = joblib.load(new_weight_root + 'rf_model_1.pkl')
gbr_model = joblib.load(new_weight_root + 'gbr_model_1.pkl')
xgboost_model = joblib.load(new_weight_root + 'xgboost_model_1.pkl')
svr_pipeline = joblib.load(new_weight_root + 'svr_pipeline_1.pkl')
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
    case_y_pred_GBR = gbr_model.predict(case_X_test)
    for col in case_X_test.select_dtypes(include=['object']).columns:
        case_X_test[col] = case_X_test[col].astype(float)
    case_y_pred_XGB = xgboost_model.predict(case_X_test)
    case_y_pred_SVR = svr_pipeline.predict(case_X_test)
    ############################################
    # # 计算当前case的MSE和 r2_score
    case_mse_RF = mean_squared_error(case_y_test, case_y_pred_RF)
    mse_scores_RF.append(case_mse_RF)
    case_mse_GBR = mean_squared_error(case_y_test, case_y_pred_GBR)
    mse_scores_GBR.append(case_mse_GBR)
    case_mse_XGB = mean_squared_error(case_y_test, case_y_pred_XGB)
    mse_scores_XGB.append(case_mse_XGB)
    case_mse_SVR = mean_squared_error(case_y_test, case_y_pred_SVR)
    mse_scores_SVR.append(case_mse_SVR)
    case_r2score_RF=r2_score(case_y_test, case_y_pred_RF)
    r2_scores_RF.append(case_r2score_RF)
    case_r2score_GBR=r2_score(case_y_test, case_y_pred_GBR)
    r2_scores_GBR.append(case_r2score_GBR)
    case_r2score_XGB=r2_score(case_y_test, case_y_pred_XGB)
    r2_scores_XGB.append(case_r2score_XGB)
    case_r2score_SVR=r2_score(case_y_test, case_y_pred_SVR)
    r2_scores_SVR.append(case_r2score_SVR)
    ############################################
    # 确保case_y_test是一个numpy数组
    case_y_test = np.array(case_y_test)
    # 对每个特征计算MSE, RMSE, 和 R^2 分数
    for i, column_name in enumerate(column_names):
        for model_name, case_y_pred in zip(['RF', 'GBR', 'XGB', 'SVR'],
                                           [case_y_pred_RF, case_y_pred_GBR, case_y_pred_XGB, case_y_pred_SVR]):
            mse = mean_squared_error(case_y_test[:, i], case_y_pred[:, i])
            r2 = r2_score(case_y_test[:, i], case_y_pred[:, i])

            metrics_dict[model_name]['MSE'].append(mse)
            metrics_dict[model_name]['RMSE'].append(np.sqrt(mse))
            metrics_dict[model_name]['R2'].append(r2)
    ############################################
    # 创建一个新的DataFrame用于存储预测值
    case_y_pred_df_RF = pd.DataFrame(case_y_pred_RF, columns=column_names)
    case_y_pred_df_GBR = pd.DataFrame(case_y_pred_GBR, columns=column_names)
    case_y_pred_df_XGB = pd.DataFrame(case_y_pred_XGB, columns=column_names)
    case_y_pred_df_SVR = pd.DataFrame(case_y_pred_SVR, columns=column_names)
    case_y_test = pd.DataFrame(case_y_test, columns=column_names)
    # 给真实值和预测值的DataFrame加上标记
    case_y_test['model'] = 'GT'
    case_y_pred_df_RF['model'] = 'RF'
    case_y_pred_df_GBR['model'] = 'GBR'
    case_y_pred_df_XGB['model'] = 'XGB'
    case_y_pred_df_SVR['model'] = 'SVR'
    # Use the index directly for timestep
    case_y_test.reset_index(drop=True, inplace=True)
    case_y_test['timestep'] = case_y_test.index
    case_y_pred_df_RF['timestep'] = case_y_pred_df_RF.index
    case_y_pred_df_GBR['timestep'] = case_y_pred_df_GBR.index
    case_y_pred_df_XGB['timestep'] = case_y_pred_df_XGB.index
    case_y_pred_df_SVR['timestep'] = case_y_pred_df_SVR.index
    # 合并真实值和预测值的DataFrame
    combined_df = pd.concat([case_y_test, case_y_pred_df_RF, case_y_pred_df_GBR, case_y_pred_df_XGB, case_y_pred_df_SVR])

    #添加caseid列
    combined_df['caseid'] = caseid
    # 重新设置索引，去除多级索引，因为我们已经有了Type列来区分数据类型
    combined_df.reset_index(drop=True, inplace=True)
    predictions.append(combined_df)  # This line was missing

# 将所有预测结果合并到一个DataFrame中
predictions_df = pd.concat(predictions)
#保存数据
predictions_df.to_csv(predata_root + 'predictions_df_machinelr_general.csv', index=False)
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
average_metrics_df.to_csv(predata_root + 'average_metrics_df_machinelr_general.csv', index=False)

# #去掉其中一个model的数据
# average_metrics_df = average_metrics_df[average_metrics_df['Model'] != 'SVR']

print(average_metrics_df)


# ########################################################################################
# 打印模型的特征重要性
feature_importances_GBR = np.mean([
    est.feature_importances_ for est in gbr_model.estimators_
], axis=0)
X_train.columns = ['age', 'sex', 'weight', 'height', 'BIS', 'PPF_CE', 'PPF_CP', 'RFTN_CE',
                'RFTN_CP', 'HR', 'MBP', 'RR', 'BT', 'PPF_VOL_single',
       'RFTN_VOL_single']
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance_RF': rf_model.feature_importances_,
    'importance_GBR': feature_importances_GBR,
    'importance_XGB': xgboost_model.estimators_[0].feature_importances_
}).sort_values('importance_RF', ascending=False)

# 更新 'feature' 列，只保留 '/' 后面的内容
feature_importances['feature'] = feature_importances['feature'].apply(lambda x: x.split('/')[-1])

print(feature_importances)
# ########################################################################################
def plot_feature_importance(feature_importances):
    # 创建DataFrame并计算累计重要性
    df = pd.DataFrame(feature_importances)
    df.set_index('feature', inplace=True)
    df['cumulative_importance'] = df.sum(axis=1)
    df.sort_values('cumulative_importance', ascending=True, inplace=True)
    # 获取排序后的索引
    sorted_idx = df.index
    # 创建堆叠柱状图
    plt.figure(figsize=(10, 6))
    # 绘制每个模型的重要性
    plt.barh(sorted_idx, df['importance_RF'], label='RF')
    plt.barh(sorted_idx, df['importance_GBR'], left=df['importance_RF'], label='GBR')
    plt.barh(sorted_idx, df['importance_XGB'], left=df['importance_RF'] + df['importance_GBR'], label='XGB')
    # 添加图例、标题和轴标签
    # plt.title('Cumulative Feature Importance for each Model')
    plt.xlabel('Cumulative Importance',fontsize=17)
    plt.ylabel('Feature',fontsize=17, labelpad=-30)
    # 调整坐标轴刻度大小
    plt.tick_params(axis='x', labelsize=13)
    plt.tick_params(axis='y', labelsize=13)
    plt.legend(loc='best', fontsize=11)
    # 调整布局
    plt.tight_layout( h_pad=None, w_pad=None, rect=[0, 0, 1, 1])  # rect参数调整左边距
    # 保存图片
    plt.savefig(image_root + 'feature_importance_general.pdf', format='pdf', bbox_inches='tight')
    # 显示图形
    plt.show()
plot_feature_importance(feature_importances)

########################################################################################
#在一张图上同时画出['model'] == 'RF'和['model'] == 'ground_truth']的所有feature的折线图，并且折线图具有置信区间的形式，并且这个折线图有两个y轴，右边的y轴显示的特征包括VENT_RR，BT，左边y轴是除了VENT_RR和BT之外的所有特征

###########################################################查看每个caseid的每个model的数据情况(ground truth)############################################################################
# predictions_df.columns = ['BIS', 'PPF20_CE', 'PPF20_CP', 'RFTN20_CE',
#                 'RFTN20_CP', 'HR', 'ART_MBP', 'VENT_RR', 'BT','model',
#          'timestep', 'caseid']
predictions_df.columns = ['BIS', 'PPF_CE', 'PPF_CP', 'RFTN_CE',
                'RFTN_CP', 'HR', 'MBP', 'RR', 'BT','model',
         'timestep', 'caseid']
# Assuming predictions_df is a previously defined DataFrame.
all_results_df = predictions_df[(predictions_df['model'] == 'RF') | (predictions_df['model'] == 'GT')]


columns_to_convert = ['BIS', 'PPF_CE', 'PPF_CP', 'RFTN_CE',
                'RFTN_CP', 'HR', 'MBP', 'RR', 'BT']
for column in columns_to_convert:
    all_results_df[column] = pd.to_numeric(all_results_df[column])

################################################################################################################################################
# 创建图和y轴figsize=(15, 8)
fig, ax = plt.subplots()
# 为每个模型和指标选择一个颜色
palette = sns.color_palette(None, len(all_results_df['model'].unique()) * len(['RR', 'BT']))
# 用于给线条命名的计数器
color_count = 0

for metric in ['RR', 'BT']:
    for model in all_results_df['model'].unique():
        # 画出每个模型的指标线条
        sns.lineplot(
            data=all_results_df[all_results_df['model'] == model],
            x='timestep',
            y=metric,
            label=f'{metric}_{model}',
            color=palette[color_count],
            estimator='mean',
            errorbar='sd',
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
# 保存图片
plt.savefig(image_root + 'ml_rr_bt_general.pdf', format='pdf', bbox_inches='tight')
plt.show()

################################################################################################################################################
# 创建图和y轴figsize=(15, 8)
fig, ax = plt.subplots()
# 为每个模型和指标选择一个颜色
palette = sns.color_palette(None, len(all_results_df['model'].unique()) * len(['PPF_CP','RFTN_CP']))
# 用于给线条命名的计数器
color_count = 0

for metric in ['PPF_CP','RFTN_CP']:
    for model in all_results_df['model'].unique():
        # 画出每个模型的指标线条
        sns.lineplot(
            data=all_results_df[all_results_df['model'] == model],
            x='timestep',
            y=metric,
            label=f'{metric}_{model}',
            color=palette[color_count],
            estimator='mean',
            errorbar='sd',
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
# 保存图片
plt.savefig(image_root + 'ml_cp_general.pdf', format='pdf', bbox_inches='tight')
plt.show()

################################################################################################################################################
# 创建图和y轴figsize=(15, 8)
fig, ax = plt.subplots()
# 为每个模型和指标选择一个颜色
palette = sns.color_palette(None,len(all_results_df['model'].unique()) * len(['BIS','HR', 'MBP']))
# 用于给线条命名的计数器
color_count = 0

for metric in ['BIS','HR', 'MBP']:
    for model in all_results_df['model'].unique():
        # 画出每个模型的指标线条
        sns.lineplot(
            data=all_results_df[all_results_df['model'] == model],
            x='timestep',
            y=metric,
            label=f'{metric}_{model}',
            color=palette[color_count],
            estimator='mean',
            errorbar='sd',
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
#保存图片
plt.savefig(image_root + 'ml_bis_hr_mbp_general.pdf', format='pdf', bbox_inches='tight')
plt.show()

################################################################################################################################################
# 创建图和y轴figsize=(15, 8)
fig, ax = plt.subplots()
# 为每个模型和指标选择一个颜色
palette = sns.color_palette(None,len(all_results_df['model'].unique()) * len(['PPF_CE','RFTN_CE']))
# 用于给线条命名的计数器
color_count = 0

for metric in ['PPF_CE','RFTN_CE']:
    for model in all_results_df['model'].unique():
        # 画出每个模型的指标线条
        sns.lineplot(
            data=all_results_df[all_results_df['model'] == model],
            x='timestep',
            y=metric,
            label=f'{metric}_{model}',
            color=palette[color_count],
            estimator='mean',
            errorbar='sd',
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
#保存图片
plt.savefig(image_root + 'ml_ce_general.pdf', format='pdf', bbox_inches='tight')
plt.show()

################################################################################################################################################










#
#
#
#
# #画出每个feature每种模型的MSE、RMSE、R^2的柱状图
# def plot_metrics(average_metrics_df, metric):
#     plt.figure(figsize=(15, 8))
#     for model in average_metrics_df['Model'].unique():
#         model_data = average_metrics_df[average_metrics_df['Model'] == model]
#         plt.bar(model_data['Feature'].unique(), model_data[model_data['Metric'] == metric]['Value'], label=model)
#     plt.title(f'{metric} for each Feature and Model')
#     plt.xlabel('Feature')
#     plt.ylabel(metric)
#     plt.legend()
#     plt.show()
#
# plot_metrics(average_metrics_df, 'MSE')
# plot_metrics(average_metrics_df, 'RMSE')
# plot_metrics(average_metrics_df, 'R2')
#
# #保存每个特征的图片，pdf格式
# plt.savefig(image_root + 'BIS_BIS_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Orchestra_PPF20_CE_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Orchestra_PPF20_CP_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Orchestra_RFTN20_CE_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Orchestra_RFTN20_CP_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Solar8000_HR_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Solar8000_ART_MBP_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Solar8000_VENT_RR_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Solar8000_BT_ML.pdf', format='pdf', bbox_inches='tight')
#
# ########################################################################################
# def convert_to_numeric(df, column_list):
#     for column in column_list:
#         df[column] = pd.to_numeric(df[column], errors='coerce')
#     return df
#
# # 计算每个时间步的平均预测值和真实值
# def mean_predictions(predictions_df):
#     # Convert relevant columns to numeric type before aggregation
#     numeric_columns =['BIS', 'PPF20_CE', 'PPF20_CP', 'RFTN20_CE',
#                 'RFTN20_CP', 'HR', 'ART_MBP', 'VENT_RR', 'BT']
#     predictions_df = convert_to_numeric(predictions_df, numeric_columns)
#
#     mean_predictions = predictions_df.groupby('timestep').agg(
#         {col: 'mean' for col in numeric_columns}
#     ).reset_index()
#     return mean_predictions
#
# #画出每个feature的所有type的折线图，并且折线图具有置信区间的形式
# def plot_feature(predictions_df, feature):
#     mean_predictions_GBR = mean_predictions(predictions_df[predictions_df['model'] == 'GBR'])
#     mean_predictions_XGB = mean_predictions(predictions_df[predictions_df['model'] == 'XGB'])
#     mean_predictions_SVR = mean_predictions(predictions_df[predictions_df['model'] == 'SVR'])
#     mean_predictions_RF = mean_predictions(predictions_df[predictions_df['model'] == 'RF'])
#     mean_predictions_ground_truth = mean_predictions(predictions_df[predictions_df['model'] == 'ground_truth'])
#
#     plt.figure(figsize=(15, 8))
#     plt.plot(mean_predictions_ground_truth['timestep'], mean_predictions_ground_truth[feature], label='True Value', color='blue')
#     plt.plot(mean_predictions_GBR['timestep'], mean_predictions_GBR[feature], label='GBR Predicted Value', color='orange')
#     plt.plot(mean_predictions_XGB['timestep'], mean_predictions_XGB[feature], label='XGB Predicted Value', color='green')
#     plt.plot(mean_predictions_SVR['timestep'], mean_predictions_SVR[feature], label='SVR Predicted Value', color='red')
#     plt.plot(mean_predictions_RF['timestep'], mean_predictions_RF[feature], label='RF Predicted Value', color='purple')
#     plt.title(f'Average True and Predicted Values over Time Steps for {feature}')
#     plt.xlabel('TimeStep')
#     plt.ylabel(feature)
#     plt.legend()
#     plt.show()
#
#
# plot_feature(predictions_df, 'BIS')
# plot_feature(predictions_df, 'PPF20_CE')
# plot_feature(predictions_df, 'PPF20_CP')
# plot_feature(predictions_df, 'RFTN20_CE')
# plot_feature(predictions_df, 'RFTN20_CP')
# plot_feature(predictions_df, 'HR')
# plot_feature(predictions_df, 'ART_MBP')
# plot_feature(predictions_df, 'VENT_RR')
# plot_feature(predictions_df, 'BT')
#
# #保存每个特征的图片，pdf格式
# plt.savefig(image_root + 'BIS_BIS_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Orchestra_PPF20_CE_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Orchestra_PPF20_CP_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Orchestra_RFTN20_CE_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Orchestra_RFTN20_CP_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Solar8000_HR_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Solar8000_ART_MBP_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Solar8000_VENT_RR_ML.pdf', format='pdf', bbox_inches='tight')
# plt.savefig(image_root + 'Solar8000_BT_ML.pdf', format='pdf', bbox_inches='tight')
########################################################################################
#在一张图上同时画出['model'] == 'RF'和['model'] == 'ground_truth']的所有feature的折线图，并且折线图具有置信区间的形式，并且这个折线图有两个y轴，左边的y轴显示的特征包括PPF20_CE，PPF20_CP，RFTN20_CE，RFTN20_CP，HR，ART_MBP，BIS右边的y轴显示的特征包括VENT_RR，BT
# def plot_feature(predictions_df, feature):
#     mean_predictions_RF = mean_predictions(predictions_df[predictions_df['model'] == 'RF'])
#     mean_predictions_ground_truth = mean_predictions(predictions_df[predictions_df['model'] == 'ground_truth'])
#
#     fig, ax1 = plt.subplots(figsize=(15, 8))
#     ax2 = ax1.twinx()
#     ax1.plot(mean_predictions_ground_truth['timestep'], mean_predictions_ground_truth[feature], label='True Value', color='blue')
#     ax1.plot(mean_predictions_RF['timestep'], mean_predictions_RF[feature], label='RF Predicted Value', color='orange')
#     ax1.set_xlabel('TimeStep')
#     ax1.set_ylabel(feature, color='black')
#     ax2.set_ylabel('VENT_RR, BT', color='black')
#     plt.title(f'Average True and Predicted Values over Time Steps for {feature}')
#     plt.legend()
#     plt.show()
#
# plot_feature(predictions_df, 'BIS')
# plot_feature(predictions_df, 'PPF20_CE')
# plot_feature(predictions_df, 'PPF20_CP')
# plot_feature(predictions_df, 'RFTN20_CE')
# plot_feature(predictions_df, 'RFTN20_CP')
# plot_feature(predictions_df, 'HR')
# plot_feature(predictions_df, 'ART_MBP')
# plot_feature(predictions_df, 'VENT_RR')
# plot_feature(predictions_df, 'BT')
# ########################################################################################
#
# # 打印模型的特征重要性
# feature_importances_GBR = np.mean([
#     est.feature_importances_ for est in gbr_model.estimators_
# ], axis=0)
#
# feature_importances = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance_RF': rf_model.feature_importances_,
#     'importance_GBR': feature_importances_GBR,
#     'importance_XGB': xgboost_model.estimators_[0].feature_importances_
# }).sort_values('importance_RF', ascending=False)
#
# print(feature_importances)
#
#
# def plot_feature_importance(feature_importances):
#     # 创建DataFrame并计算累计重要性
#     df = pd.DataFrame(feature_importances)
#     df.set_index('feature', inplace=True)
#     df['cumulative_importance'] = df.sum(axis=1)
#     df.sort_values('cumulative_importance', ascending=True, inplace=True)
#     # 获取排序后的索引
#     sorted_idx = df.index
#     # 创建堆叠柱状图
#     plt.figure(figsize=(15, 8))
#     # 绘制每个模型的重要性
#     plt.barh(sorted_idx, df['importance_RF'], label='RF')
#     plt.barh(sorted_idx, df['importance_GBR'], left=df['importance_RF'], label='GBR')
#     plt.barh(sorted_idx, df['importance_XGB'], left=df['importance_RF'] + df['importance_GBR'], label='XGB')
#     # 添加图例、标题和轴标签
#     # plt.title('Cumulative Feature Importance for each Model')
#     plt.xlabel('Cumulative Importance')
#     plt.ylabel('Feature')
#     plt.legend()
#     # 显示图形
#     plt.show()
# plot_feature_importance(feature_importances)
# #
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# #####################################################RandomForestRegressor########################################################
# # # 将所有数据用于训练模型
# # X_train, y_train = data_process(trajectory_train_general)
# # # 创建随机森林回归模型
# # rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# #
# # # 训练模型
# # rf_model.fit(X_train, y_train)
# #
# # # 打印模型的特征重要性
# # feature_importances = pd.DataFrame({
# #     'feature': X_train.columns,
# #     'importance': rf_model.feature_importances_
# # }).sort_values('importance', ascending=False)
# #
# # print(feature_importances)
# # joblib.dump(rf_model, new_weight_root + 'rf_model.pkl')
# # print('rf_model saved')
# #
#
#
# # 加载模型
# rf_model = joblib.load(new_weight_root + 'rf_model.pkl')
#
# # 初始化一个空的列表来存储每个case的MSE
# mse_scores = []
# # 分组交叉验证
# predictions = []
# for caseid, case_data in trajectory_test_general.groupby('caseid'):
#     # 提取当前case的测试数据
#     case_X_test, case_y_test = data_process(case_data)
#     # 进行预测
#     case_y_pred = rf_model.predict(case_X_test)
#     # 存储预测结果和真实值
#     # 计算当前case的MSE
#     case_mse = mean_squared_error(case_y_test, case_y_pred)
#     mse_scores.append(case_mse)
#     # column_names 列表包含所有预期的列名，应当与 case_y_test 的列数匹配
#     column_names = ['BIS/BIS', 'Orchestra/PPF20_CE', 'Orchestra/PPF20_CP', 'Orchestra/RFTN20_CE',
#                     'Orchestra/RFTN20_CP', 'Solar8000/HR', 'Solar8000/ART_MBP', 'Solar8000/VENT_RR', 'Solar8000/BT']
#
#     # case_y_test 已经是一个DataFrame，确保它的列名与column_names一致
#     case_y_test.columns = column_names
#
#     # 创建一个新的DataFrame用于存储预测值
#     case_y_pred_df = pd.DataFrame(case_y_pred, columns=column_names)
#
#     # 给真实值和预测值的DataFrame加上标记
#     case_y_test['Type'] = 'ground_truth'
#     case_y_pred_df['Type'] = 'RF_Predictions'
#
#     # Use the index directly for timestep
#     case_y_test['timestep'] = case_y_test.index
#     case_y_pred_df['timestep'] = case_y_pred_df.index
#     # 合并真实值和预测值的DataFrame
#     combined_df = pd.concat([case_y_test, case_y_pred_df])
#
#     #添加caseid列
#     combined_df['caseid'] = caseid
#     # 重新设置索引，去除多级索引，因为我们已经有了Type列来区分数据类型
#     combined_df.reset_index(drop=True, inplace=True)
#     predictions.append(combined_df)  # This line was missing
#
# # 将所有预测结果合并到一个DataFrame中
# predictions_df = pd.concat(predictions)
#
# #画出真实值和预测值的所有列的所有case的对比图，不同的模型数据用type列区分
#
#
#
#
#
# # 计算平均MSE
# average_mse = np.mean(mse_scores)
# print(f"Average MSE_rf_model: {average_mse}")
# # 计算RMSE
# average_rmse = np.sqrt(average_mse)
# print(f"Average RMSE_rf_model: {average_rmse}")
#
# # 计算每个时间步的平均预测值和真实值
# mean_predictions = predictions_df.groupby('timestep').agg({'y_true': 'mean', 'y_pred': 'mean'}).reset_index()
#
# # 计算置信区间
# confidence_intervals = predictions_df.groupby('timestep')['y_pred'].agg(
#     lambda x: stats.norm.interval(0.95, loc=np.mean(x), scale=stats.sem(x)))
#
# # 将置信区间拆分为上下界
# mean_predictions['y_pred_ci_lower'] = [ci[0] for ci in confidence_intervals]
# mean_predictions['y_pred_ci_upper'] = [ci[1] for ci in confidence_intervals]
#
# # 可视化
# plt.figure(figsize=(15, 8))
# plt.plot(mean_predictions['timestep'], mean_predictions['y_true'], label='True Value', color='blue')
# plt.plot(mean_predictions['timestep'], mean_predictions['y_pred'], label='Predicted Value', color='orange')
#
# # 绘制置信区间
# plt.fill_between(mean_predictions['timestep'], mean_predictions['y_pred_ci_lower'], mean_predictions['y_pred_ci_upper'],
#                  color='orange', alpha=0.2)
#
# plt.title('Average True and Predicted Values over Time Steps with Confidence Intervals')
# plt.xlabel('TimeStep')
# plt.ylabel('MSE')
# plt.legend()
# plt.show()
#
#
#
# mse = mean_squared_error(predictions_df['y_true'], predictions_df['y_pred'])
# r2 = r2_score(predictions_df['y_true'], predictions_df['y_pred'])
# print(f'Mean Squared Error_rf_model: {mse}')
# print(f'R^2——rf_model: {r2}')
#
#
#
# #####################################################RandomForestRegressor########################################################
#
# #####################################################GradientBoostingRegressor########################################################
#
#
# # 创建梯度提升回归模型
# gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
#                                       max_depth=3, random_state=42)
#
# # 训练模型
# gbr_model.fit(X_train, y_train)
#
# # 初始化一个空的列表来存储每个case的MSE
# mse_scores = []
# # 分组交叉验证
# predictions = []
# for caseid, case_data in trajectory_test_general.groupby('caseid'):
#     # 提取当前case的测试数据
#     case_X_test = case_data[['state_dl', 'Orchestra/PPF20_VOL_single', 'Orchestra/RFTN20_VOL_single']]
#     case_y_test = case_data['next_state_dl_env']
#     # 进行预测
#     case_y_pred = gbr_model.predict(case_X_test)
#
#     # 存储预测结果和真实值
#     predictions.append(pd.DataFrame({
#         'timestep': case_data['timestep'],
#         'y_true': case_y_test,
#         'y_pred': case_y_pred
#     }))
#     # 计算当前case的MSE
#     case_mse = mean_squared_error(case_y_test, case_y_pred)
#     mse_scores.append(case_mse)
# # 将所有预测结果合并到一个DataFrame中
# predictions_df = pd.concat(predictions)
# # 计算平均MSE
# average_mse = np.mean(mse_scores)
# print(f"Average MSE_gbr_model: {average_mse}")
# # 计算RMSE
# average_rmse = np.sqrt(average_mse)
# print(f"Average RMSE_gbr_model: {average_rmse}")
#
# # 计算每个时间步的平均预测值和真实值
# mean_predictions = predictions_df.groupby('timestep').agg({'y_true': 'mean', 'y_pred': 'mean'}).reset_index()
#
# # 计算置信区间
# confidence_intervals = predictions_df.groupby('timestep')['y_pred'].agg(
#     lambda x: stats.norm.interval(0.95, loc=np.mean(x), scale=stats.sem(x)))
#
# # 将置信区间拆分为上下界
# mean_predictions['y_pred_ci_lower'] = [ci[0] for ci in confidence_intervals]
# mean_predictions['y_pred_ci_upper'] = [ci[1] for ci in confidence_intervals]
#
# # 可视化
# plt.figure(figsize=(15, 8))
# plt.plot(mean_predictions['timestep'], mean_predictions['y_true'], label='True Value', color='blue')
# plt.plot(mean_predictions['timestep'], mean_predictions['y_pred'], label='Predicted Value', color='orange')
#
# # 绘制置信区间
# plt.fill_between(mean_predictions['timestep'], mean_predictions['y_pred_ci_lower'], mean_predictions['y_pred_ci_upper'],
#                  color='orange', alpha=0.2)
#
# plt.title('Average True and Predicted Values over Time Steps with Confidence Intervals')
# plt.xlabel('TimeStep')
# plt.ylabel('BIS')
# plt.legend()
# plt.show()
#
# # 打印模型的特征重要性
# feature_importances = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance': gbr_model.feature_importances_
# }).sort_values('importance', ascending=False)
#
# print(feature_importances)
#
# #打印metrics
# from sklearn.metrics import mean_squared_error, r2_score
# mse = mean_squared_error(predictions_df['y_true'], predictions_df['y_pred'])
# r2 = r2_score(predictions_df['y_true'], predictions_df['y_pred'])
# print(f'Mean Squared Error_gbr_model: {mse}')
# print(f'R^2_gbr_model: {r2}')
#
#
# joblib.dump(gbr_model, new_weight_root + 'gbr_model.pkl')
# print('gbr_model saved')
#
#
# #####################################################xgboost########################################################
# # 将所有数据用于训练模型
#
# # 创建XGBoost回归模型
# xgboost_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
#
# # 训练模型
# xgboost_model.fit(X_train, y_train)
#
# # 初始化一个空的列表来存储每个case的MSE
# mse_scores = []
# # 分组交叉验证
# predictions = []
# for caseid, case_data in trajectory_test_general.groupby('caseid'):
#     # 提取当前case的测试数据
#     case_X_test = case_data[['state_dl', 'Orchestra/PPF20_VOL_single', 'Orchestra/RFTN20_VOL_single']]
#     case_y_test = case_data['next_state_dl_env']
#     # 进行预测
#     case_y_pred = xgboost_model.predict(case_X_test)
#
#     # 存储预测结果和真实值
#     predictions.append(pd.DataFrame({
#         'timestep': case_data['timestep'],
#         'y_true': case_y_test,
#         'y_pred': case_y_pred
#     }))
#     # 计算当前case的MSE
#     case_mse = mean_squared_error(case_y_test, case_y_pred)
#     mse_scores.append(case_mse)
# # 将所有预测结果合并到一个DataFrame中
# predictions_df = pd.concat(predictions)
# # 计算平均MSE
# average_mse = np.mean(mse_scores)
# print(f"Average MSE_xgboost_model: {average_mse}")
# # 计算RMSE
# average_rmse = np.sqrt(average_mse)
# print(f"Average RMSE_xgboost_model: {average_rmse}")
#
# # 计算每个时间步的平均预测值和真实值
# mean_predictions = predictions_df.groupby('timestep').agg({'y_true': 'mean', 'y_pred': 'mean'}).reset_index()
#
# # 计算置信区间
# confidence_intervals = predictions_df.groupby('timestep')['y_pred'].agg(
#     lambda x: stats.norm.interval(0.95, loc=np.mean(x), scale=stats.sem(x)))
#
# # 将置信区间拆分为上下界
# mean_predictions['y_pred_ci_lower'] = [ci[0] for ci in confidence_intervals]
# mean_predictions['y_pred_ci_upper'] = [ci[1] for ci in confidence_intervals]
#
# # 可视化
# plt.figure(figsize=(15, 8))
# plt.plot(mean_predictions['timestep'], mean_predictions['y_true'], label='True Value', color='blue')
# plt.plot(mean_predictions['timestep'], mean_predictions['y_pred'], label='Predicted Value', color='orange')
#
# # 绘制置信区间
# plt.fill_between(mean_predictions['timestep'], mean_predictions['y_pred_ci_lower'], mean_predictions['y_pred_ci_upper'],
#                  color='orange', alpha=0.2)
#
# plt.title('Average True and Predicted Values over Time Steps with Confidence Intervals')
# plt.xlabel('TimeStep')
# plt.ylabel('BIS')
# plt.legend()
# plt.show()
#
# # 打印模型的特征重要性
# feature_importances = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance': xgboost_model.feature_importances_
# }).sort_values('importance', ascending=False)
#
# print(feature_importances)
#
# #打印metrics
# mse = mean_squared_error(predictions_df['y_true'], predictions_df['y_pred'])
# r2 = r2_score(predictions_df['y_true'], predictions_df['y_pred'])
# print(f'Mean Squared Error_xgboost_model: {mse}')
# print(f'R^2_xgboost_model: {r2}')
#
#
# joblib.dump(xgboost_model, new_weight_root + 'xgboost_model.pkl')
# print('xgboost_model saved')
#
#
# #####################################################支持向量回归（SVR）########################################################
# # 创建SVR模型的pipeline，包括预处理步骤
# svr_pipeline = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
# # 训练模型
# svr_pipeline.fit(X_train, y_train)
#
# # 分组交叉验证
# predictions = []
# for caseid, case_data in trajectory_test_general.groupby('caseid'):
#     # 提取当前case的测试数据
#     case_X_test = case_data[['state_dl', 'Orchestra/PPF20_VOL_single', 'Orchestra/RFTN20_VOL_single']]
#     case_y_test = case_data['next_state_dl_env']
#
#     # 进行预测
#     case_y_pred = svr_pipeline.predict(case_X_test)
#
#     # 存储预测结果和真实值
#     predictions.append(pd.DataFrame({
#         'timestep': case_data['timestep'],
#         'y_true': case_y_test,
#         'y_pred': case_y_pred
#     }))
#
#     # 计算当前case的MSE
#     case_mse = mean_squared_error(case_y_test, case_y_pred)
#     mse_scores.append(case_mse)
# # 将所有预测结果合并到一个DataFrame中
# predictions_df = pd.concat(predictions)
# # 计算平均MSE
# average_mse = np.mean(mse_scores)
# print(f"Average MSE_svr: {average_mse}")
# # 计算RMSE
# average_rmse = np.sqrt(average_mse)
# print(f"Average RMSE_svr: {average_rmse}")
# # 计算每个时间步的平均预测值和真实值
# mean_predictions = predictions_df.groupby('timestep').agg({'y_true': 'mean', 'y_pred': 'mean'}).reset_index()
#
# # 计算置信区间
# confidence_intervals = predictions_df.groupby('timestep')['y_pred'].agg(
#     lambda x: stats.norm.interval(0.95, loc=np.mean(x), scale=stats.sem(x)))
#
# # 将置信区间拆分为上下界
# mean_predictions['y_pred_ci_lower'] = [ci[0] for ci in confidence_intervals]
# mean_predictions['y_pred_ci_upper'] = [ci[1] for ci in confidence_intervals]
#
# # 可视化
# plt.figure(figsize=(15, 8))
# plt.plot(mean_predictions['timestep'], mean_predictions['y_true'], label='True Value', color='blue')
# plt.plot(mean_predictions['timestep'], mean_predictions['y_pred'], label='Predicted Value', color='orange')
#
# # 绘制置信区间
# plt.fill_between(mean_predictions['timestep'], mean_predictions['y_pred_ci_lower'], mean_predictions['y_pred_ci_upper'],
#                  color='orange', alpha=0.2)
#
# plt.title('Average True and Predicted Values over Time Steps with Confidence Intervals')
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
#
#
# # # 打印模型的特征重要性
# # feature_importances = pd.DataFrame({
# #     'feature': X_train.columns,
# #     'importance': svr_pipeline.feature_importances_
# # }).sort_values('importance', ascending=False)
# #
# # print(feature_importances)
#
# #打印metrics
# mse = mean_squared_error(predictions_df['y_true'], predictions_df['y_pred'])
# r2 = r2_score(predictions_df['y_true'], predictions_df['y_pred'])
# print(f'Mean Squared Error_svr: {mse}')
# print(f'R^2_svr: {r2}')
#
#
# joblib.dump(svr_pipeline, new_weight_root + 'svr_pipeline.pkl')
# print('svr_pipeline saved')
#
# #####################################################支持向量回归（SVR）########################################################
