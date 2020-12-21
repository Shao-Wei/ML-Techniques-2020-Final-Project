import numpy as np
import pandas as pd
import math
import preprocessing as pp
import util
import train
import xgboost as xgb
import csv

# preprocessing
print("start preprocessing ...")
dfTrain, dfRawTrain, dfAdr, dfIsCancelled, dfAdrReal, feature_train = pp.preprocessing_train('../data/train.csv')
dfTest, dfRawTest = pp.preprocessing_test('../data/test.csv', feature_train)
print("preprocessing done ...")

print("start training ..")
# adr, is_canceled combined
modelList_adr_real = train.train_adr(dfTrain, dfAdrReal, 5)
# adr, is_canceled separated
# modelList_adr = train.train_adr(dfTrain, dfAdr, 5)
# modelList_isCancelded = train.train_isCanceled(dfTrain, dfIsCancelled, 5)
print("training done ...")

print("computing revenue ...")
# adr, is_canceled combined
revenue_train = train.predict_revenue_ensemble_adr_isCanceled_combined(dfTrain, dfRawTrain, modelList_adr_real, 1)
revenue_test = train.predict_revenue_ensemble_adr_isCanceled_combined(dfTest, dfRawTest, modelList_adr_real, 0)
# adr, is_canceled separated
# revenue_train = train.predict_revenue_ensemble_adr_isCanceled_separated(dfTrain, dfRawTrain, modelList_adr, modelList_isCancelded, 1)
# revenue_test = train.predict_revenue_ensemble_adr_isCanceled_separated(dfTest, dfRawTest, modelList_adr, modelList_isCancelded, 0)
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