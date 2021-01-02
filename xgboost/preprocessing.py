import numpy as np
import pandas as pd
import util

del_features = ['ID',
                'arrival_date_year', 
                'arrival_date_month', 
                'arrival_date_week_number', 
                'arrival_date_day_of_month', 
                'country', 
                'agent', 
                'company', 
                ]

label_features = ['reservation_status', 
                  'reservation_status_date', 
                  'is_canceled',
                  'adr',
                  ]

stand_features = ['lead_time', 
                  'stays_in_weekend_nights', 
                  'stays_in_week_nights', 
                  'adults', 
                  'children', 
                  'babies', 
                  'previous_cancellations', 
                  'previous_bookings_not_canceled', 
                  'booking_changes', 
                  'days_in_waiting_list', 
                  'required_car_parking_spaces', 
                  'total_of_special_requests'
                  ]

one_hot_list = ['hotel', 
                'meal', 
                'market_segment', 
                'distribution_channel', 
                'reserved_room_type', 
                'assigned_room_type', 
                'deposit_type', 
                'customer_type'
                ]

# encode_list = [['arrival_date_month', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],        # 0: feature, 1~len-1: types
#                ]

# makeNaN_list = [['company', -1],      # 0: feature, 1: label
#                 ['agent', -1],
#                 ['children', 0],   # 'children' in train.csv has missing entries
#                 ]

# makeOthers_list = [['country', 'PRT', 'GBR', 'other'],        # 0: feature, 1~len-2: types, len-1: label
#                    #['market_segment', 'Online TA', 'Offline TA/TO', 'other'],
#                    #['distribution_channel', 'TA/TO', 'Direct', 'other'],
#                    ['agent', 9, -1, -2],
#                    ['company', -1, 40, -2],
#                    ]

def feature_delete_one(df, feature):
    del df[feature]

def feature_delete(df, features):
    for i in range(len(features)):
        feature_delete_one(df, features[i])

def feature_standardize_one(df, feature):
    max = df[feature].max()
    min = df[feature].min()
    df[feature] = (df[feature]-min)/(max-min)

def feature_standardize(df, features):
    for i in range(len(features)):
        feature_standardize_one(df, features[i])

# def feature_makeNaN_one(df, feature, label):
#     for i in range(len(df)):
#         if (pd.isnull(df.iloc[i][feature])):
#             df.iat[i, df.columns.get_loc(feature)] = label

# def feature_makeNaN(df, list):
#     for i in range(len(list)):
#         feature_makeNaN_one(df, list[i][0], list[i][1])

def checkIsNull(df):
    if df.isnull().values.any():
        features = list(df.columns[:]) # get list of all features
        for target in features:
            if df[target].isnull().values.any():
                df[target] = df[target].replace(np.nan, 0)
    return df

# def feature_makeOthers_one(df, feature, types, label):
#     for i in range(len(df)):
#         isOther = 1
#         for j in range(len(types)):
#             isOther = isOther & (df.iloc[i][feature] != types[j])
#             # if (isOther == 0): break
#         if (isOther):
#             df.iat[i, df.columns.get_loc(feature)] = label

# def feature_makeOthers(df, list):
#     for i in range(len(list)):
#         one_list = list[i]
#         feature_makeOthers_one(df, one_list[0], one_list[1:len(one_list) - 1], one_list[len(one_list) - 1])
        
def feature_one_hot_one(df, feature):
    pOHE = pd.get_dummies(df[feature], prefix = feature)
    df.drop(feature, inplace=True, axis = 1)
    dfNew = pd.concat([df, pOHE], axis = 1)
    return dfNew

def feature_one_hot(df, list):
    dfNew = df.copy()
    for target in list:
        dfNew = feature_one_hot_one(dfNew, target)
    return dfNew

# def feature_encode_one(df, feature, types):
#     column = []
#     for i in range(len(df)):
#         for j in range(len(types)):
#             if (df.iloc[i][feature] == types[j]):
#                 column.append(j)
#                 continue
    
#     df.insert(len(df.columns), feature, column, True) 
#     feature_delete_one(df, feature)

# def feature_encode(df, list):
#     for i in range(len(list)):
#         one_list = list[i]
#         feature_encode_one(df, one_list[0], one_list[1:len(one_list)])

# def feature_merge_two_multi(df, feature1, feature2, name): # feature = (feature1)' * feature2
#     column = []
#     for i in range(len(df)):
#         column.append((not df.iloc[i][feature1]) * df.iloc[i][feature2])
    
#     df.insert(0, name, column, True) 
#     feature_delete_one(df, feature1)
#     feature_delete_one(df, feature2)

def preprocessing(file_train, file_test):
    dfRawTrain = pd.read_csv(file_train)
    dfRawTrain = dfRawTrain.sample(frac=1) # shuffle
    dfRawTrain.reset_index(drop=True, inplace=True)
    dfTrain = dfRawTrain.copy()

    dfRawTest = pd.read_csv(file_test)
    dfTest = dfRawTest.copy()

    dfIsCanceled = dfTrain['is_canceled'].to_frame() # reserve label is_canceled
    dfIsCanceled.reset_index(drop=True, inplace=True)
    dfAdr = dfTrain['adr'].to_frame() # reserve label adr
    dfAdr.reset_index(drop=True, inplace=True)
    dfIsCanceled = checkIsNull(dfIsCanceled)
    dfAdr = checkIsNull(dfAdr)
    dfAdrReal = ((dfIsCanceled['is_canceled'] == 0) * dfAdr['adr']).to_frame() # get real adr
    dfAdrReal = dfAdrReal.rename(columns={0: 'adr_real'})

    feature_delete(dfTrain, label_features)
    dfAll = pd.concat([dfTrain, dfTest])
    nTrain = len(dfTrain)
    feature_delete(dfAll, del_features)
    dfAll = checkIsNull(dfAll)
    dfAll = feature_one_hot(dfAll, one_hot_list)
    feature_standardize(dfAll, stand_features)
    #data_train = data_train.dropna()
    dfAll = dfAll.sort_index(ascending = False, axis = 1)
    dfTrain = dfAll.iloc[:nTrain, :]
    dfTest = dfAll.iloc[nTrain:,  :]

    return dfTrain, dfRawTrain, dfAdr, dfIsCanceled, dfAdrReal, dfTest, dfRawTest

# def preprocessing_revenue():
#     data_train = pd.read_csv('../data/train.csv')
#     data_train_label = pd.read_csv('../data/train_label.csv')
#     feature_delete_one(data_train_label, 'arrival_date')
#     revenue_no_quan = util.get_revenue_no_quan(data_train)
#     data_train_label.insert(len(data_train_label.columns), 'label_no_quan', revenue_no_quan, True) 
#     data_train_label.to_csv('../data/train_label_label_no_quan.csv', index=False, header=False)
