import numpy as np
import pandas as pd
import math
import preprocessing as pp
import xgboost as xgb
import csv

# preprocessing
data_train = pd.read_csv('../data/train.csv')

pp.feature_delete(data_train, pp.del_features)
pp.feature_standardize(data_train, pp.stand_features)
pp.feature_makeNaN(data_train, pp.makeNaN_features)
pp.feature_makeOthers(data_train, pp.makeOthers_list)
pp.feature_one_hot(data_train, pp.one_hot_list)
pp.feature_merge_two_multi(data_train, 'is_canceled', 'adr', 'adr_real')

data_train = data_train.dropna()
data_train = data_train.sample(frac=1) # shuffle
data_train.to_csv('../data/train_pre.csv', index=False)