import numpy as np
import pandas as pd
import preprocessing as pp
import training as tr
import util

## path declaration
pTraincsv = "../data/train.csv"
pTrainLabel = '../data/train_label.csv'
pTestcsv = "../data/test.csv"

## get data, alias
print("Start preprocessing ...")
dfTrain, dfAdr, dfIsCanceled, dfAdrReal, dfRawTrain, feature_train = pp.preprocess_train(pTraincsv) # dfRawTrain used in deriving revenue 
dfTest, dfRawTest = pp.preprocess_test(pTestcsv, feature_train)
print("Preprocessing done ...")

print("Start training ..")
modelList_adr_real = tr.train_adr(dfTrain, dfAdrReal, 5)
print("Training done ...")

print("Computing revenue ...")
revenue_train = util.predict_revenue_ensemble_adr_isCanceled_combined(dfTrain, dfRawTrain, modelList_adr_real, 1)
# revenue_test = util.predict_revenue_ensemble_adr_isCanceled_combined(dfTest, dfRawTest, modelList_adr_real, 0)
dfTrainLabel = pd.read_csv(pTrainLabel)
npTrainLabel = dfTrainLabel['label'].tolist()
Ein_L1 = util.L1_Error(npTrainLabel, revenue_train)

print("  Ein_L1 = ", Ein_L1)