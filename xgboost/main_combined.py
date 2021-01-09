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
dfTrain, dfRawTrain, dfAdr, dfIsCancelled, dfAdrReal, dfTest, dfRawTest = pp.preprocessing('../data/train.csv', '../data/test.csv')
# dfTrain.to_csv('dfTrain.csv', index=False)
# dfTest.to_csv('dfTest.csv', index=False)
print("preprocessing done ...")

print("start training ..")
# adr, is_canceled combined
modelList_adr_real, dtrain_adr_real, dvalid_adr_real = train.train_adr(dfTrain, dfAdrReal, 5, 0.8, 500)
print("training done ...")

# compute error
Ein_adr_real = util.rmse(dtrain_adr_real.get_label(), train.predict_adr_ensemble(dtrain_adr_real, modelList_adr_real))
Eval_adr_real = util.rmse(dvalid_adr_real.get_label(), train.predict_adr_ensemble(dvalid_adr_real, modelList_adr_real))
print("Ein_adr_real: ", Ein_adr_real)
print("Eval_adr_real: ", Eval_adr_real)


print("computing revenue ...")
# adr, is_canceled combined
revenue_train = train.predict_revenue_ensemble_adr_isCanceled_combined(dfTrain, dfRawTrain, modelList_adr_real, 1)
revenue_test = train.predict_revenue_ensemble_adr_isCanceled_combined(dfTest, dfRawTest, modelList_adr_real, 0)
print("train_predict: ", revenue_train)
print("computing revenue done ...")

dfTrain_label = pd.read_csv('../data/train_label.csv')
train_label = dfTrain_label['label'].tolist()
print("train_label: ", train_label)
Ein_0_1 = util.zero_one_Error(train_label, revenue_train)
Ein_L1 = util.L1_Error(train_label, revenue_train)

print("Ein_0_1 = ", Ein_0_1)
print("Ein_L1 = ", Ein_L1)
print("test_pridict: ", revenue_test)

# write predict
testlabel_filename = sys.argv[1]
dfTest_nolabel = pd.read_csv('../data/test_nolabel.csv')
dfTest_nolabel['label'] = revenue_test
dfTest_nolabel.to_csv(testlabel_filename, index=False)
