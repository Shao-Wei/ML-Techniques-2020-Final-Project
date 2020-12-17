import numpy as np
import pandas as pd
import preprocessing as pp
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import svm

## path declaration
traincsv = "../data/train.csv"
testcsv = "../data/test.csv"

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

## get data
dfRawTrain, dfAdr, dfRawTest = pp.preprocess(traincsv, testcsv)

y_train = dfAdr
X_train = dfRawTrain

# for d in range(5, 40, 4): # Best: d = 13, mean = 0.41, w/o cv = 0.72 
#     print("max_depth: ", d)
#     dt = DecisionTreeRegressor(max_depth=d, random_state=99)
#     dt_fit = dt.fit(X_train, y_train)
#     dt_scores = cross_val_score(dt_fit, X_train, y_train, cv = 10)
#     print("    mean cross validation score: ", np.mean(dt_scores))
#     print("    score without cv: ", dt_fit.score(X_train, y_train))

dt = DecisionTreeRegressor(max_depth=13, random_state=99)
dt_fit = dt.fit(X_train, y_train)
dt_scores = cross_val_score(dt_fit, X_train, y_train, cv = 5)
print("mean cross validation score: ", np.mean(dt_scores))
print("cross validation score: ", dt_scores)