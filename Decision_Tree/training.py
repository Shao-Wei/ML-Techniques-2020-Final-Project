import numpy as np
import pandas as pd
import math
import preprocessing as pp
import util

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import cross_val_score

## classifiers
classifiers = [
    svm.SVR(),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
    linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()
]

## training
def train_adr(dfTrain, dfLabel, k):
    #algs = [[5, 20], [8, 20], [11, 20]]
    algs = [[8, 20]]
    nAlg = len(algs)
    modelList_all = [[] for i in range(nAlg)]
    E_algs = [0] * nAlg

    for i in range(k):
        print(i, "- fold", sep = ' ')
        # split train df
        nTrain = dfTrain.shape[0]
        iBegin = int(dfTrain.shape[0]*(1/k)*i)
        iEnd = int(dfTrain.shape[0]*(1/k)*(i+1))
        idxList = [i for i in range(iBegin, iEnd)]

        dfTrainValid = dfTrain.take(idxList, axis = 0)
        dfTrainPart = dfTrain.drop(idxList, axis = 0, inplace = False)
        dfLabelValid = dfLabel.take(idxList, axis = 0)
        dfLabelPart = dfLabel.drop(idxList, axis = 0, inplace = False)
        # cv on  each parameter setting
        for j in range(nAlg):
            print("    Training ..", algs[j])
            ft = RandomForestRegressor(max_depth=algs[j][0], min_samples_split = algs[j][1], random_state=99)
            ft_fit = ft.fit(dfTrainPart, np.ravel(dfLabelPart))
            npPredict = ft.predict(dfTrainValid)
            modelList_all[j].append(ft)
            E_algs[j] = E_algs[j] + util.rmse(np.ravel(dfLabelValid), npPredict)
    
    E_algs = [x/k for x in E_algs]
    bestSetting = util.get_bestAlg(E_algs)
    print("Best setting: ", algs[bestSetting], sep = ' ')
    model = modelList_all[bestSetting]
    return model

def train_is_canceled(dfTrain, dfLabel, k):
    #algs = [[5, 20], [8, 20], [11, 20]]
    algs = [[8, 20]]
    nAlg = len(algs)
    modelList_all = [[] for i in range(nAlg)]
    E_algs = [0] * nAlg

    for i in range(k):
        print(i, "- fold", sep = ' ')
        # split train df
        nTrain = dfTrain.shape[0]
        iBegin = int(dfTrain.shape[0]*(1/k)*i)
        iEnd = int(dfTrain.shape[0]*(1/k)*(i+1))
        idxList = [i for i in range(iBegin, iEnd)]

        dfTrainValid = dfTrain.take(idxList, axis = 0)
        dfTrainPart = dfTrain.drop(idxList, axis = 0, inplace = False)
        dfLabelValid = dfLabel.take(idxList, axis = 0)
        dfLabelPart = dfLabel.drop(idxList, axis = 0, inplace = False)
        # cv on  each parameter setting
        for j in range(nAlg):
            print("    Training ..", algs[j])
            ft = DecisionTreeClassifier(max_depth=algs[j][0], min_samples_split = algs[j][1], random_state=99)
            ft_fit = ft.fit(dfTrainPart, np.ravel(dfLabelPart))
            npPredict = ft.predict(dfTrainValid)
            modelList_all[j].append(ft)
            E_algs[j] = E_algs[j] + util.zero_one_Error(np.ravel(dfLabelValid), npPredict)
    
    E_algs = [x/k for x in E_algs]
    bestSetting = util.get_bestAlg(E_algs)
    print("Best setting: ", algs[bestSetting], sep = ' ')
    model = modelList_all[bestSetting]
    return model