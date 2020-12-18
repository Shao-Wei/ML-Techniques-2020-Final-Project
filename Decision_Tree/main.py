import numpy as np
import pandas as pd
import preprocessing as pp
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import cross_val_score

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
dfTrain, dfAdr, feature_train = pp.preprocess_train(traincsv)
dfTest = pp.preprocess_test(testcsv, feature_train)

y_train = dfAdr
y_1darr = np.ravel(y_train)
X_train = dfTrain

# for d in range(5, 40, 4): # Best: d = 9, mean = 0.39, w/o cv = 0.60
#     print("max_depth: ", d)
#     dt = DecisionTreeRegressor(max_depth=d, min_samples_split = 20, random_state=99) # inforcing min_samples_split gives better cv score
#     dt_fit = dt.fit(X_train, y_train)
#     dt_scores = cross_val_score(dt_fit, X_train, y_train, scoring='r2', cv=5)
#     print("    mean cross validation score: ", np.mean(dt_scores))
#     print("    score without cv: ", dt_fit.score(X_train, y_train))

# for d in range(5, 40, 4): # Best: d = 13, mean = 0.46, w/o cv = 0.76 
#     print("max_depth: ", d)
#     ft = RandomForestRegressor(max_depth=d, min_samples_split = 20, random_state=99)
#     ft_fit = ft.fit(X_train, y_1darr)
#     ft_scores = cross_val_score(ft_fit, X_train, y_1darr, scoring='r2', cv = 5)
#     print("    mean cross validation score: ", np.mean(ft_scores))
#     print("    score without cv: ", ft_fit.score(X_train, y_train))

ft = RandomForestRegressor(max_depth=13, min_samples_split = 20, random_state=99)
ft_fit = ft.fit(X_train, y_1darr)
npPredictResults = ft.predict(dfTrain)
dfPredictResults = pd.DataFrame({'Predicted': npPredictResults[:]})
dfPredictResults['adr'] = dfAdr
dfPredictResults.to_csv('predict.csv')