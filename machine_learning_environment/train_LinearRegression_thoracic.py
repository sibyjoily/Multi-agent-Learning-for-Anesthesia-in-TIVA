from common.data_processing_test import proceesing
from common.read_yaml import ReadYaml as read_yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
#打印metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
#保存模型
import joblib
from sklearn.model_selection import GridSearchCV

param = read_yaml()
param = param.read()
address_param = param['basic_config']
data_root = address_param['data_root']
predata_root = address_param['predata_root']
new_weight_root= address_param['weight_root']

device = "cuda"
process = proceesing()

# 加载数据
trajectory_train_general = pd.read_csv(predata_root + 'trajectory_train_Thoracic4.csv')
trajectory_test_general = pd.read_csv(predata_root + 'trajectory_test_Thoracic4.csv')
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

# 将所有数据用于训练模型
X_train, y_train = data_process(trajectory_train_general)

#####################################################RandomForestRegressor########################################################

# #####################################################RandomForestRegressor########################################################
# # 创建随机森林回归模型 Best parameters: {'bootstrap': False, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2}
rf_model = RandomForestRegressor(n_estimators=100,max_depth=30, min_samples_split=2, min_samples_leaf=1, max_features= 'sqrt', bootstrap=False)
# 训练模型
rf_model.fit(X_train, y_train)

# 打印模型的特征重要性
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
# rf_model_thoracic.pkl
print(feature_importances)
joblib.dump(rf_model, new_weight_root + 'rf_thoracic_2.pkl')
print('rf_model saved')
# #
###########################################################################################################
# ####################################################GradientBoostingRegressor########################################################
# # # 创建梯度提升回归模型
# gbr_model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
#                                       max_depth=3, random_state=42))
# # 训练模型
# gbr_model.fit(X_train, y_train)
# # 打印模型的特征重要性
# feature_importances = np.mean([
#     est.feature_importances_ for est in gbr_model.estimators_
# ], axis=0)
# feature_importances_df = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance': feature_importances
# }).sort_values('importance', ascending=False)
#
# joblib.dump(gbr_model, new_weight_root + 'gbr_model_thoracic_1.pkl')
# print('gbr_model saved')
# ####################################################GradientBoostingRegressor########################################################
#
# ####################################################xgboost########################################################
# for col in X_train.select_dtypes(include=['object']).columns:
#     X_train[col] = X_train[col].astype(float)
# # 创建XGBoost回归模型
# xgboost_model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
# # 训练模型
# xgboost_model.fit(X_train, y_train)
# # Save the trained model to a file
# joblib.dump(xgboost_model, new_weight_root + 'xgboost_model_thoracic_1.pkl')
# print('xgboost_model saved')
# # 打印模型的特征重要性
# feature_importances = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance': xgboost_model.estimators_[0].feature_importances_
# }).sort_values('importance', ascending=False)
#
# print(feature_importances)
#
# #####################################################xgboost########################################################
# #####################################################支持向量回归（SVR）########################################################
# # Create SVR model's pipeline, including preprocessing steps
# svr_pipeline = make_pipeline(
#     StandardScaler(),
#     MultiOutputRegressor(SVR(C=1.0, epsilon=0.2))
# )
# # 训练模型
# svr_pipeline.fit(X_train, y_train)
#
# joblib.dump(svr_pipeline, new_weight_root + 'svr_pipeline_thoracic_1.pkl')
# print('svr_pipeline saved')
# ####################################################支持向量回归（SVR）########################################################
#
#
#


















# ###############################模型预测############################################
#
# #####################################################加载模型########################################################
# # 加载模型
# rf_model = joblib.load(new_weight_root + 'rf_model.pkl')
# gbr_model = joblib.load(new_weight_root + 'gbr_model.pkl')
# xgboost_model = joblib.load(new_weight_root + 'xgboost_model.pkl')
# svr_pipeline = joblib.load(new_weight_root + 'svr_pipeline.pkl')
# #####################################################加载模型########################################################
# # 初始化一个空的列表来存储每个case的MSE
# mse_scores_RF = []
# mse_scores_GBR = []
# mse_scores_XGB = []
# mse_scores_SVR = []
# # 分组交叉验证
# predictions = []
# ########################################################################################
# for caseid, case_data in trajectory_test_general.groupby('caseid'):
#     # 提取当前case的测试数据
#     case_X_test, case_y_test = data_process(case_data)
#     # 进行预测
#     case_y_pred_RF = rf_model.predict(case_X_test)
#     case_y_pred_GBR = gbr_model.predict(case_X_test)
#     case_y_pred_XGB = xgboost_model.predict(case_X_test)
#     case_y_pred_SVR = svr_pipeline.predict(case_X_test)
#     # 计算当前case的MSE
#     case_mse_RF = mean_squared_error(case_y_test, case_y_pred_RF)
#     mse_scores_RF.append(case_mse_RF)
#     case_mse_GBR = mean_squared_error(case_y_test, case_y_pred_GBR)
#     mse_scores_GBR.append(case_mse_GBR)
#     case_mse_XGB = mean_squared_error(case_y_test, case_y_pred_XGB)
#     mse_scores_XGB.append(case_mse_XGB)
#     case_mse_SVR = mean_squared_error(case_y_test, case_y_pred_SVR)
#     mse_scores_SVR.append(case_mse_SVR)
#     # column_names 列表包含所有预期的列名，应当与 case_y_test 的列数匹配
#     column_names = ['BIS/BIS', 'Orchestra/PPF20_CE', 'Orchestra/PPF20_CP', 'Orchestra/RFTN20_CE',
#                     'Orchestra/RFTN20_CP', 'Solar8000/HR', 'Solar8000/ART_MBP', 'Solar8000/VENT_RR', 'Solar8000/BT']
#
#     # case_y_test 已经是一个DataFrame，确保它的列名与column_names一致
#     case_y_test.columns = column_names
#
#     # 创建一个新的DataFrame用于存储预测值
#     case_y_pred_df_RF = pd.DataFrame(case_y_pred_RF, columns=column_names)
#     case_y_pred_df_GBR = pd.DataFrame(case_y_pred_GBR, columns=column_names)
#     case_y_pred_df_XGB = pd.DataFrame(case_y_pred_XGB, columns=column_names)
#     case_y_pred_df_SVR = pd.DataFrame(case_y_pred_SVR, columns=column_names)
#     # 给真实值和预测值的DataFrame加上标记
#     case_y_test['Type'] = 'ground_truth'
#     case_y_pred_df_RF['Type'] = 'RF_Predictions'
#     case_y_pred_df_GBR['Type'] = 'GBR_Predictions'
#     case_y_pred_df_XGB['Type'] = 'XGB_Predictions'
#     case_y_pred_df_SVR['Type'] = 'SVR_Predictions'
#     # Use the index directly for timestep
#     case_y_test['timestep'] = case_y_test.index
#     case_y_pred_df_RF['timestep'] = case_y_pred_df_RF.index
#     case_y_pred_df_GBR['timestep'] = case_y_pred_df_GBR.index
#     case_y_pred_df_XGB['timestep'] = case_y_pred_df_XGB.index
#     case_y_pred_df_SVR['timestep'] = case_y_pred_df_SVR.index
#     # 合并真实值和预测值的DataFrame
#     combined_df = pd.concat([case_y_test, case_y_pred_df_RF, case_y_pred_df_GBR, case_y_pred_df_XGB, case_y_pred_df_SVR])
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
#
# # 计算每种model的平均MSE
# average_mse_RF = np.mean(mse_scores_RF)
# average_mse_GBR = np.mean(mse_scores_GBR)
# average_mse_XGB = np.mean(mse_scores_XGB)
# average_mse_SVR = np.mean(mse_scores_SVR)
# print(f"Average MSE_rf_model: {average_mse_RF}")
# print(f"Average MSE_gbr_model: {average_mse_GBR}")
# print(f"Average MSE_xgboost_model: {average_mse_XGB}")
# print(f"Average MSE_svr: {average_mse_SVR}")
# # 计算RMSE
# average_rmse_RF = np.sqrt(average_mse_RF)
# average_rmse_GBR = np.sqrt(average_mse_GBR)
# average_rmse_XGB = np.sqrt(average_mse_XGB)
# average_rmse_SVR = np.sqrt(average_mse_SVR)
# print(f"Average RMSE_rf_model: {average_rmse_RF}")
# print(f"Average RMSE_gbr_model: {average_rmse_GBR}")
# print(f"Average RMSE_xgboost_model: {average_rmse_XGB}")
# print(f"Average RMSE_svr: {average_rmse_SVR}")
#
# # 计算每个时间步的平均预测值和真实值
# def mean_predictions(predictions_df):
#     mean_predictions = predictions_df.groupby('timestep').agg(
#         {'BIS/BIS': 'mean', 'Orchestra/PPF20_CE': 'mean', 'Orchestra/PPF20_CP': 'mean', 'Orchestra/RFTN20_CE': 'mean',
#          'Orchestra/RFTN20_CP': 'mean', 'Solar8000/HR': 'mean', 'Solar8000/ART_MBP': 'mean', 'Solar8000/VENT_RR': 'mean',
#          'Solar8000/BT': 'mean'}).reset_index()
#     return mean_predictions
# mean_predictions_GBR = mean_predictions(predictions_df[predictions_df['Type'] == 'GBR_Predictions'])
# mean_predictions_XGB = mean_predictions(predictions_df[predictions_df['Type'] == 'XGB_Predictions'])
# mean_predictions_SVR = mean_predictions(predictions_df[predictions_df['Type'] == 'SVR_Predictions'])
# mean_predictions_RF = mean_predictions(predictions_df[predictions_df['Type'] == 'RF_Predictions'])
# mean_predictions_ground_truth = mean_predictions(predictions_df[predictions_df['Type'] == 'ground_truth'])
#
# #计算每个model与case_y_test['Type'] = 'ground_truth'的所有feature的MSE
# mse_GBR = mean_squared_error(mean_predictions_GBR.iloc[:, 1:], mean_predictions_ground_truth.iloc[:, 1:])
# mse_XGB = mean_squared_error(mean_predictions_XGB.iloc[:, 1:], mean_predictions_ground_truth.iloc[:, 1:])
# mse_SVR = mean_squared_error(mean_predictions_SVR.iloc[:, 1:], mean_predictions_ground_truth.iloc[:, 1:])
# mse_RF = mean_squared_error(mean_predictions_RF.iloc[:, 1:], mean_predictions_ground_truth.iloc[:, 1:])
# print(f"MSE_rf_model: {mse_RF}")
# print(f"MSE_gbr_model: {mse_GBR}")
# print(f"MSE_xgboost_model: {mse_XGB}")
# print(f"MSE_svr: {mse_SVR}")
# # 计算R^2
# r2_GBR = r2_score(mean_predictions_GBR.iloc[:, 1:], mean_predictions_ground_truth.iloc[:, 1:])
# r2_XGB = r2_score(mean_predictions_XGB.iloc[:, 1:], mean_predictions_ground_truth.iloc[:, 1:])
# r2_SVR = r2_score(mean_predictions_SVR.iloc[:, 1:], mean_predictions_ground_truth.iloc[:, 1:])
# r2_RF = r2_score(mean_predictions_RF.iloc[:, 1:], mean_predictions_ground_truth.iloc[:, 1:])
# print(f"R^2_rf_model: {r2_RF}")
# print(f"R^2_gbr_model: {r2_GBR}")
# print(f"R^2_xgboost_model: {r2_XGB}")
# print(f"R^2_svr: {r2_SVR}")
# #画出每个feature的所有type的折线图，并且折线图具有置信区间的形式
# def plot_feature(predictions_df, feature):
#     mean_predictions_GBR = mean_predictions(predictions_df[predictions_df['Type'] == 'GBR_Predictions'])
#     mean_predictions_XGB = mean_predictions(predictions_df[predictions_df['Type'] == 'XGB_Predictions'])
#     mean_predictions_SVR = mean_predictions(predictions_df[predictions_df['Type'] == 'SVR_Predictions'])
#     mean_predictions_RF = mean_predictions(predictions_df[predictions_df['Type'] == 'RF_Predictions'])
#     mean_predictions_ground_truth = mean_predictions(predictions_df[predictions_df['Type'] == 'ground_truth'])
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
# plot_feature(predictions_df, 'BIS/BIS')
# plot_feature(predictions_df, 'Orchestra/PPF20_CE')
# plot_feature(predictions_df, 'Orchestra/PPF20_CP')
# plot_feature(predictions_df, 'Orchestra/RFTN20_CE')
# plot_feature(predictions_df, 'Orchestra/RFTN20_CP')
# plot_feature(predictions_df, 'Solar8000/HR')
# plot_feature(predictions_df, 'Solar8000/ART_MBP')
# plot_feature(predictions_df, 'Solar8000/VENT_RR')
# plot_feature(predictions_df, 'Solar8000/BT')
#
# ################################################################################################################
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
# joblib.dump(gbr_model, new_weight_root + 'gbr_model.pkl')
# print('gbr_model saved')
#
# #打印metrics
# mse = mean_squared_error(predictions_df['y_true'], predictions_df['y_pred'])
# r2 = r2_score(predictions_df['y_true'], predictions_df['y_pred'])
# print(f'Mean Squared Error_gbr_model: {mse}')
# print(f'R^2_gbr_model: {r2}')
#
#
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
# # 打印模型的特征重要性
# feature_importances = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance': xgboost_model.feature_importances_
# }).sort_values('importance', ascending=False)
#
# print(feature_importances)
# joblib.dump(xgboost_model, new_weight_root + 'xgboost_model.pkl')
# print('xgboost_model saved')
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
#
# #打印metrics
# mse = mean_squared_error(predictions_df['y_true'], predictions_df['y_pred'])
# r2 = r2_score(predictions_df['y_true'], predictions_df['y_pred'])
# print(f'Mean Squared Error_xgboost_model: {mse}')
# print(f'R^2_xgboost_model: {r2}')
#
#
#
#
# #####################################################支持向量回归（SVR）########################################################
# # 创建SVR模型的pipeline，包括预处理步骤
# svr_pipeline = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
# # 训练模型
# svr_pipeline.fit(X_train, y_train)
#
# joblib.dump(svr_pipeline, new_weight_root + 'svr_pipeline.pkl')
# print('svr_pipeline saved')
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
#
# #####################################################支持向量回归（SVR）########################################################
#
# from sklearn.model_selection import GridSearchCV
#
# # 定义参数网格
# param_grid = {
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt'],
#     'bootstrap': [True, False]
# }
# # # 创建随机森林回归模型
# rf_model = RandomForestRegressor(n_estimators=100)
# # 创建 GridSearchCV 对象
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)
#
# # 打印最佳参数
# print("Best parameters:", grid_search.best_params_)