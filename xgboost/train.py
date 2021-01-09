import numpy as np
import pandas as pd
import math
import preprocessing as pp
import util
import xgboost as xgb
from scipy import stats
import csv

def train_adr(dfTrain, dfLabel, k, train_percent, num_round):
    print("start training adr ...")
    dtrain_all = xgb.DMatrix(dfTrain, label=dfLabel)
    nX = dtrain_all.num_row()
    nX_train = int(train_percent*nX)
    trainInd = list(range(0, nX_train))
    validInd = list(range(nX_train, nX))
    dtrain = dtrain_all.slice(trainInd)
    dvalid = dtrain_all.slice(validInd)

    fold = util.getFold(k, nX_train)

    param_name = ['max_depth', 'eta']
    algs = [[10, 0.3]] # [max_depth, eta]
    nAlg = len(algs)
    modelList_all = [[] for _ in range(nAlg)]
    E_algs = [0] * nAlg
    for i in range(k):
        train_out_Ind = fold[i]
        train_in_Ind = []
        for j in range(k):
            if (j != i):
                train_in_Ind = train_in_Ind + fold[j]
        dtrain_in = dtrain.slice(train_in_Ind)
        dtrain_out = dtrain.slice(train_out_Ind)
    
        for j in range(nAlg):
            watchlist = [(dtrain_in,'train'), (dtrain_out,'val')]
            param = {'max_depth': algs[j][0], 'eta': algs[j][1], 'objective':'reg:squarederror'}
            bst = xgb.train(param, dtrain_in, num_boost_round = num_round, early_stopping_rounds = 10, evals = watchlist)
            preds = bst.predict(dtrain_out)
            modelList_all[j].append(bst)
            E_algs[j] = E_algs[j] + util.rmse(dtrain_out.get_label(), preds)

    E_algs[:] = [x/k for x in E_algs]
    iBestAlg = util.get_iBestAlg(E_algs, param_name, algs)
    modelList = modelList_all[iBestAlg]
    #param = {'max_depth': bestAlg[0], 'eta': bestAlg[1], 'objective':'reg:squarederror'}
    #bst = xgb.train(param, data_adr, num_round)
    #adr_preds = bst.predict(data_adr_noshuffle)

    return modelList, dtrain, dvalid

def train_isCanceled(dfTrain, dfLabel, k, train_percent, num_round):
    print("start training is_canceled ...")
    dtrain_all = xgb.DMatrix(dfTrain, label=dfLabel)
    nX = dtrain_all.num_row()
    nX_train = int(train_percent*nX)
    trainInd = list(range(0, nX_train))
    validInd = list(range(nX_train, nX))
    dtrain = dtrain_all.slice(trainInd)
    dvalid = dtrain_all.slice(validInd)

    fold = util.getFold(k, nX_train)

    param_name = ['max_depth', 'eta']
    algs = [[10, 0.3]] # [max_depth, eta]
    nAlg = len(algs)
    modelList_all = [[] for _ in range(nAlg)]
    E_algs = [0] * nAlg
    for i in range(k):
        train_out_Ind = fold[i]
        train_in_Ind = []
        for j in range(k):
            if (j != i):
                train_in_Ind = train_in_Ind + fold[j]
        dtrain_in = dtrain.slice(train_in_Ind)
        dtrain_out = dtrain.slice(train_out_Ind)
    
        for j in range(nAlg):
            watchlist = [(dtrain_in,'train'), (dtrain_out,'val')]
            param = {'max_depth': algs[j][0], 'eta': algs[j][1], 'objective':'multi:softmax',  'num_class': 2, 'eval_metric':'merror'}
            bst = xgb.train(param, dtrain_in, num_boost_round = num_round, early_stopping_rounds = 10, evals = watchlist)
            preds = bst.predict(dtrain_out)
            modelList_all[j].append(bst)
            E_algs[j] = E_algs[j] + util.zero_one_Error(dtrain_out.get_label(), preds)

    E_algs[:] = [x/k for x in E_algs]
    iBestAlg = util.get_iBestAlg(E_algs, param_name, algs)
    modelList = modelList_all[iBestAlg]
    #param = {'max_depth': bestAlg[0], 'eta': bestAlg[1], 'objective':'reg:squarederror'}
    #bst = xgb.train(param, data_adr, num_round)
    #adr_preds = bst.predict(data_adr_noshuffle)

    return modelList, dtrain, dvalid

def predict_adr_ensemble(data, modelList):
    preds = [0] * data.num_row()
    for i in range(len(modelList)):
        preds = preds + modelList[i].predict(data)
    preds = preds / len(modelList)
    return preds

def predict_isCanceled_ensemble(data, modelList):
    preds_list = [[] for _ in range(len(modelList))]
    for i in range(len(modelList)):
        preds_list[i] = modelList[i].predict(data)

    preds_list = np.transpose(np.array(preds_list))
    preds = []
    for list in preds_list:
        preds = preds + [stats.mode(list)[0][0]]

    return preds

# uniform aggregation
def predict_revenue_ensemble_adr_isCanceled_combined(df, dfRaw, modelList, isTrain):
    data = xgb.DMatrix(df, label=None) # label is not used
    preds = predict_adr_ensemble(data, modelList)
    # print(np.mean(np.array(preds)))
    # dict = {'adr': preds}
    # df_temp = pd.DataFrame(dict)
    # if (isTrain):
    #     df_temp.to_csv('adr_train.csv')
    # else:
    #     df_temp.to_csv('adr_test.csv')

    if (isTrain):
        revenue = get_revenue_train(dfRaw, preds)
    else:
        revenue = get_revenue_test(dfRaw, preds)
    revenue = revenue_quantization(revenue)
    return revenue
    # return 0

# uniform aggregation
def predict_revenue_ensemble_adr_isCanceled_separated(df, dfRaw, modelList_adr, modelList_isCancelded, isTrain):
    data = xgb.DMatrix(df, label=None) # label is not used
    preds_adr = predict_adr_ensemble(data, modelList_adr)
    preds_isCancelded = predict_isCanceled_ensemble(data, modelList_isCancelded)

    preds = np.multiply(np.logical_not(preds_isCancelded), preds_adr)
    if (isTrain):
        revenue = get_revenue_train(dfRaw, preds)
    else:
        revenue = get_revenue_test(dfRaw, preds)
    revenue = revenue_quantization(revenue)
    return revenue

month_dic = {
    'January': 0,
    'February': 1,
    'March': 2,
    'April': 3,
    'May': 4, 
    'June': 5, 
    'July': 6, 
    'August': 7, 
    'September': 8, 
    'October': 9, 
    'November': 10, 
    'December': 11
}

def delete_non_date_train(list):
    list_out = list
    list_out[0][1][28:31] = []
    list_out[0][3][30:31] = []
    list_out[0][5][30:31] = []
    list_out[0][8][30:31] = []
    list_out[0][10][30:31] = []
    list_out[1][1][29:31] = []
    list_out[1][3][30:31] = []
    list_out[1][5][30:31] = []
    list_out[1][8][30:31] = []
    list_out[1][10][30:31] = []
    list_out[2][1][28:31] = []
    list_out[2][3][30:31] = []
    list_out[2][5][30:31] = []
    list_out[2][8][30:31] = []
    list_out[2][10][30:31] = []
    list_out[0][0:6] = [[]]
    list_out[2][3:12] = [[]]
    return list_out

def delete_non_date_test(list):
    list_out = list
    list_out[0][30:31] = []
    list_out[2][30:31] = []
    return list_out

def get_revenue_train(df, adr_preds):
    revenue = [[[] for _ in range(12)] for _ in range(3)]
    for i in range(3):
        for j in range(12):
            revenue[i][j] = [0]*31
    for i in range(len(df)):
        revenue[df.iloc[i]['arrival_date_year']-2015][month_dic[df.iloc[i]['arrival_date_month']]][df.iloc[i]['arrival_date_day_of_month']-1] += (df.iloc[i]['stays_in_weekend_nights']+df.iloc[i]['stays_in_week_nights'])*adr_preds[i]
    revenue = delete_non_date_train(revenue)
    return [c for a in revenue for b in a for c in b]

def get_revenue_test(df, adr_preds):
    revenue = [[] for _ in range(5)]
    for i in range(5):
        revenue[i] = [0]*31
    for i in range(len(df)):
        revenue[month_dic[df.iloc[i]['arrival_date_month']]-3][df.iloc[i]['arrival_date_day_of_month']-1] += (df.iloc[i]['stays_in_weekend_nights']+df.iloc[i]['stays_in_week_nights'])*adr_preds[i]
    revenue = delete_non_date_test(revenue)
    return [b for a in revenue for b in a]

def revenue_quantization(revenue):
    revenue_quan = [int(math.floor(x/10000)) for x in revenue]
    #revenue_quan = [0 if x < 0 else x for x in revenue_quan] # set negative values to zero
    return revenue_quan