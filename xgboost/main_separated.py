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
print("preprocessing done ...")

print("start training ..")
# adr, is_canceled separated
modelList_adr, dtrain_adr, dvalid_adr = train.train_adr(dfTrain, dfAdr, 5, 0.8, 500)
modelList_isCancelded, dtrain_isCancelded, dvalid_isCancelded = train.train_isCanceled(dfTrain, dfIsCancelled, 5, 0.8, 500)
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
revenue_train = train.predict_revenue_ensemble_adr_isCanceled_separated(dfTrain, dfRawTrain, modelList_adr, modelList_isCancelded, 1)
revenue_test = train.predict_revenue_ensemble_adr_isCanceled_separated(dfTest, dfRawTest, modelList_adr, modelList_isCancelded, 0)
print("train_pridict: ", revenue_train)
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