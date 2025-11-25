
from dev2.train_multi_agent import MultiAgentPolicyBased

from dev2.train_single_agent import SingleAgentPolicyBased
# #gbr_general_1 rf_general_2 xgboost_general_1 QPLEX  CW_QMIX  general  rf_general_2
#VDN,QMIX，CW_QMIX__online thoracic already  All data test done
#MADDPG,COMA,VDN,QMIX
# # ############################################################################################################
# model = MultiAgentPolicyBased(model_name='QMIX', dataset_name='general', is_sampling=True,
#                               scheme='offline', test_env_model_name='rf_general_2')


# model.test(reload=True,is_sampling=False,dataset_name='general')
############################################################################################################
# # # # # #DQN，DuelingDQN，DDPG,PPO
model = SingleAgentPolicyBased(model_name='DuelingDQN', dataset_name='general', is_sampling=True,
                              scheme='offline', test_env_model_name='rf_general_2')
# #
# model.test(reload=True,is_sampling=False,dataset_name='general')
# ###########################################################################################################
#
model.run()
print('run done')
model.save()
model.draw()

model.test(reload=False,is_sampling=True,dataset_name='general')

print('done')





