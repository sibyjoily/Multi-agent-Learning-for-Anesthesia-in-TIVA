from common.data_processing_test import proceesing
from common.read_yaml import ReadYaml as read_yaml
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
#保存模型
import joblib

param = read_yaml()
param = param.read()
address_param = param['basic_config']
data_root = address_param['data_root']
predata_root = address_param['predata_root']
new_weight_root= address_param['weight_root']

device = "cuda"
process = proceesing()

# 加载数据
trajectory_train_general = pd.read_csv(predata_root + 'trajectory_train_general4.csv')
trajectory_test_general = pd.read_csv(predata_root + 'trajectory_test_general4.csv')
#将数据合并成一个
# trajectory_general = pd.concat([trajectory_train_general, trajectory_test_general], axis=0)
#############################
def data_process(data):
    X_1 = data['state_dl']
    X_1 = X_1.str.split(',', expand=True)
    X_1.columns = ['age', 'sex', 'weight', 'height','BIS/BIS','Orchestra/PPF20_CE', 'Orchestra/PPF20_CP','Orchestra/RFTN20_CE','Orchestra/RFTN20_CP',
                                               'Solar8000/HR','Solar8000/ART_MBP','Solar8000/VENT_RR','Solar8000/BT']
    X_2 = data[['Orchestra/PPF20_VOL_single', 'Orchestra/RFTN20_VOL_single']]
    X = pd.concat([X_1, X_2], axis=1)
    y = data['next_state_dl_env']
    y = y.str.split(',', expand=True)
    y.columns = ['BIS/BIS', 'Orchestra/PPF20_CE', 'Orchestra/PPF20_CP', 'Orchestra/RFTN20_CE',
                'Orchestra/RFTN20_CP', 'Solar8000/HR', 'Solar8000/ART_MBP', 'Solar8000/VENT_RR', 'Solar8000/BT']
    return X, y

# 将所有数据用于训练模型
X_train, y_train = data_process(trajectory_train_general)
# # #####################################################RandomForestRegressor########################################################
# 创建随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100,max_depth=30, min_samples_split=2, min_samples_leaf=1, max_features= 'sqrt', bootstrap=False)
# 训练模型
rf_model.fit(X_train, y_train)
# 打印模型的特征重要性
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importances)
joblib.dump(rf_model, new_weight_root + 'rf_general_2.pkl')
print('rf_model saved')

###########################################################################################################
# # ####################################################GradientBoostingRegressor########################################################
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
# joblib.dump(gbr_model, new_weight_root + 'gbr_general_1.pkl')
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
# joblib.dump(xgboost_model, new_weight_root + 'xgboost_general_1.pkl')
# print('xgboost_model saved')
# # 打印模型的特征重要性
# feature_importances = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance': xgboost_model.estimators_[0].feature_importances_
# }).sort_values('importance', ascending=False)
#
# print(feature_importances)