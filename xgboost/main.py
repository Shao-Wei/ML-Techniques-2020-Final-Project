import numpy as np
import pandas as pd
import math
import preprocessing as pp
import util
import xgboost as xgb
import csv

# preprocessing
print("start preprocessing ...")
#pp.preprocessing_adr()
#pp.preprocessing_revenue()
print("preprocessing done ...")

# train adr
print("start training adr ...")
data_adr = xgb.DMatrix('../data/train_pre.csv?format=csv&label_column=0')
data_adr_noshuffle = xgb.DMatrix('../data/train_pre_noshuffle.csv?format=csv&label_column=0')
k = 5 # k-fold
nX = data_adr.num_row()
nValid = int(data_adr.num_row()*(1/k))
fold = [[] for _ in range(k)]
for i in range(k):
    if (i == k - 1):
        fold[i] = fold[i] + list(range(nValid*i, nX))
    else:
        fold[i] = fold[i] + list(range(nValid*i, nValid*(i+1)))

algs = [[3, 1], [4, 1], [5, 1], [6, 1], [7, 1]] # [max_depth, eta]
nAlg = len(algs)
E_algs = [0] * nAlg
num_round = 10
for i in range(k):
    validInd = fold[i]
    trainInd = []
    for j in range(k):
        if (j != i):
            trainInd = trainInd + fold[j]
    dtrain = data_adr.slice(trainInd)
    dvalid = data_adr.slice(validInd)
    
    for j in range(nAlg):
        param = {'max_depth': algs[j][0], 'eta': algs[j][1], 'objective':'reg:squarederror'}
        bst = xgb.train(param, dtrain, num_round)
        preds = bst.predict(dvalid)
        E_algs[j] = E_algs[j] + util.squaredError(dvalid.get_label(), preds)

E_algs[:] = [x/k for x in E_algs]
bestAlg = util.get_bestAlg(algs, E_algs)
param = {'max_depth': bestAlg[0], 'eta': bestAlg[1], 'objective':'reg:squarederror'}
bst = xgb.train(param, data_adr, num_round)
adr_preds = bst.predict(data_adr_noshuffle)

# calculate revenue from adr_preds
data_train = pd.read_csv('../data/train.csv')
revenue_predict = util.get_revenue_predict(data_train, adr_preds)
print("training adr done ...")

# train quantization
print("start training quantization ...")
data_quan = xgb.DMatrix('../data/train_label_label_no_quan.csv?format=csv&label_column=0')
k = 1 # k-fold
nX = data_quan.num_row()
nValid = int(data_quan.num_row()*(1/k))
fold = [[] for _ in range(k)]
for i in range(k):
    if (i == k - 1):
        fold[i] = fold[i] + list(range(nValid*i, nX))
    else:
        fold[i] = fold[i] + list(range(nValid*i, nValid*(i+1)))

algs = [[3, 1], [4, 1], [5, 1], [6, 1], [7, 1]] # [max_depth, eta]
nAlg = len(algs)
E_algs = [0] * nAlg
num_round = 10
for i in range(k):
    validInd = fold[i]
    trainInd = []
    for j in range(k):
        if (j != i):
            trainInd = trainInd + fold[j]
    #dtrain = data_quan.slice(trainInd) # k > 1 
    dtrain = data_quan.slice(validInd) # k = 1: use all training data
    dvalid = data_quan.slice(validInd)
    #num_class = np.amax(dtrain.get_label()) - np.amin(dtrain.get_label()) + 1
    
    for j in range(nAlg):
        param = {'max_depth': algs[j][0], 'eta': algs[j][1], 'objective':'multi:softmax',  'num_class': 10, 'eval_metric':'mae'}
        bst = xgb.train(param, dtrain, num_round)
        preds = bst.predict(dvalid)
        E_algs[j] = E_algs[j] + util.L1_Error(dvalid.get_label(), preds)

E_algs[:] = [x/k for x in E_algs]
bestAlg = util.get_bestAlg(algs, E_algs)
param = {'max_depth': bestAlg[0], 'eta': bestAlg[1], 'objective':'multi:softmax',  'num_class': 10, 'eval_metric':'mae'}
bst = xgb.train(param, data_quan, num_round)
print("training quantization done ...")

# calculate final error
data_error = pd.read_csv('../data/train_label.csv')
pp.feature_delete_one(data_error, 'arrival_date')
data_error.insert(len(data_error.columns), 'label_predict', revenue_predict, True) 
data_error.to_csv('../data/train_label_predict.csv', index=False, header=False)
data_error = xgb.DMatrix('../data/train_label_predict.csv?format=csv&label_column=0')
preds = bst.predict(data_error)
E = util.L1_Error(data_error.get_label(), preds)
print("final E_train = ", E, sep='')