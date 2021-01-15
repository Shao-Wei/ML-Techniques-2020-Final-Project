import numpy as np
import pandas as pd
import math
import preprocessing as pp
import util
import train
import xgboost as xgb
import csv
import sys

# preprocessing
print("start preprocessing ...")
dfTrain, dfRawTrain, dfAdr_train, dfIsCanceled_train, dfAdrReal_train, dfValid, dfRawValid, dfAdr_valid, dfIsCanceled_valid, dfAdrReal_valid, dfTest, dfRawTest = pp.preprocessing('../data/train.csv', '../data/test.csv')
print("preprocessing done ...")

print("start training ..")
# adr, is_canceled separated
modelList_adr, dtrain_adr, dvalid_adr = train.train_adr(dfTrain, dfAdr_train, dfValid, dfAdr_valid, 5, 10000)
modelList_isCancelded, dtrain_isCancelded, dvalid_isCancelded = train.train_isCanceled(dfTrain, dfIsCanceled_train, dfValid, dfIsCanceled_valid, 5, 10000)
print("training done ...")

# compute error
Ein_adr = util.rmse(dtrain_adr.get_label(), train.predict_adr_ensemble(dtrain_adr, modelList_adr))
Eval_adr = util.rmse(dvalid_adr.get_label(), train.predict_adr_ensemble(dvalid_adr, modelList_adr))
Ein_isCancelded = util.zero_one_Error(dtrain_isCancelded.get_label(), train.predict_isCanceled_ensemble(dtrain_isCancelded, modelList_isCancelded))
Eval_isCancelded = util.zero_one_Error(dvalid_isCancelded.get_label(), train.predict_isCanceled_ensemble(dvalid_isCancelded, modelList_isCancelded))
print("Ein_adr: ", Ein_adr)
print("Eval_adr: ", Eval_adr)
print("Ein_isCancelded: ", Ein_isCancelded)
print("Eval_isCancelded: ", Eval_isCancelded)


print("computing revenue ...")
# adr, is_canceled separated
revenue_train = train.predict_revenue_ensemble_adr_isCanceled_separated(dfTrain, dfRawTrain, modelList_adr, modelList_isCancelded, 0)
revenue_valid = train.predict_revenue_ensemble_adr_isCanceled_separated(dfValid, dfRawValid, modelList_adr, modelList_isCancelded, 1)
revenue_test = train.predict_revenue_ensemble_adr_isCanceled_separated(dfTest, dfRawTest, modelList_adr, modelList_isCancelded, 2)
# print("train_pridict: ", revenue_train)
print("computing revenue done ...")

dfTrain_label = pd.read_csv('../data/train_label.csv')
train_label = dfTrain_label['label'].tolist()
# print("train_label: ", train_label)
revenue_all = (np.array(revenue_train) + np.array(revenue_valid)).tolist()
# for valid set 1
# revenue_train = revenue_all[129:]
# revenue_valid = revenue_all[:129]
# valid_label = train_label[:129]
# train_label = train_label[129:]
# # for valid set 2
# revenue_train = revenue_all[:129] + revenue_all[256:]
# revenue_valid = revenue_all[129:256]
# valid_label = train_label[129:256]
# train_label = train_label[:129] + train_label[256:]
# # for valid set 3
# revenue_train = revenue_all[:256] + revenue_all[384:]
# revenue_valid = revenue_all[256:384]
# valid_label = train_label[256:384]
# train_label = train_label[:256] + train_label[384:]
# # for valid set 4
# revenue_train = revenue_all[:384] + revenue_all[511:]
# revenue_valid = revenue_all[384:511]
# valid_label = train_label[384:511]
# train_label = train_label[:384] + train_label[511:]
# for valid set 5
revenue_train = revenue_all[:511]
revenue_valid = revenue_all[511:]
valid_label = train_label[511:]
train_label = train_label[:511]

# train quantization
revenue_train, revenue_valid, revenue_test = train.train_quantize(revenue_train, train_label, revenue_valid, valid_label, revenue_test, 10000)

#Ein_0_1 = util.zero_one_Error(train_label, revenue_train)
#Ein_L1 = util.L1_Error(train_label, revenue_train)
#Eval_0_1 = util.zero_one_Error(valid_label, revenue_valid)
Eval_L1 = util.L1_Error(valid_label, revenue_valid)

#print("Ein_0_1 = ", Ein_0_1)
#print("Ein_L1 = ", Ein_L1)
#print("Eval_0_1 = ", Eval_0_1)
print("Eval_L1 = ", Eval_L1)
print("test_pridict: ", revenue_test)

# write predict
testlabel_filename = sys.argv[1]
dfTest_nolabel = pd.read_csv('../data/test_nolabel.csv')
dfTest_nolabel['label'] = revenue_test
dfTest_nolabel.to_csv(testlabel_filename, index=False)