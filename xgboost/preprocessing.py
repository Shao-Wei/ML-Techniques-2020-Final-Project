import numpy as np
import pandas as pd
import util

del_features = ['ID',
                'arrival_date_year', 
                # 'arrival_date_month', 
                'arrival_date_week_number', 
                # 'arrival_date_day_of_month', 
                'country', 
                'agent', 
                'company', 
                ]
                
del_corr_features_ohe = [ # corr < 0.05
                         'reserved_room_type_P',
                         'reserved_room_type_L',
                         'reserved_room_type_B',
                         'meal_Undefined',
                         'meal_SC',
                         'meal_FB',
                         'market_segment_Undefined',
                         'market_segment_Aviation',
                         'distribution_channel_Undefined',
                         'distribution_channel_GDS',
                         'deposit_type_Refundable',
                         'customer_type_Group',
                         'customer_type_Contract',
                         'assigned_room_type_P',
                         'assigned_room_type_L',
                         'assigned_room_type_K',
                         'assigned_room_type_B'
                         # corr < 0.1
                         # 'reserved_room_type_H',
                         # 'reserved_room_type_E',
                         # 'reserved_room_type_C',
                         # 'meal_HB',
                         # 'meal_BB',
                         # 'market_segment_Corporate',
                         # 'assigned_room_type_I',
                         # 'assigned_room_type_H',
                         # 'assigned_room_type_E',
                         # 'assigned_room_type_C'
                         ]
                         
del_corr_features_stand = [ # corr < 0.05
                            # corr < 0.1
                            'stays_in_weekend_nights',
                            'previous_bookings_not_canceled',
                            'days_in_waiting_list',
                            'babies',
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
                'arrival_date_month',
                # 'arrival_date_week_number',  
                'arrival_date_day_of_month', 
                'meal', 
                'market_segment', 
                'distribution_channel', 
                'reserved_room_type', 
                'assigned_room_type', 
                'deposit_type', 
                'customer_type'
                ]

pow_list_for_stand = ['total_of_special_requests',
                      'required_car_parking_spaces',
                      'lead_time',
                      'children',
                      ]
multi_2_list_for_one_hot = ['reserved_room_type_A',
                            'market_segment_Online TA',
                            'market_segment_Groups',
                            'market_segment_Direct',
                            'hotel_Resort Hotel',
                            'hotel_City Hotel',
                            'deposit_type_Non Refund',
                            'deposit_type_No Deposit',
                            'assigned_room_type_A'
                            ]

stand_list_for_one_hot = ['reserved_room_type_P',
                          'reserved_room_type_L',
                          'reserved_room_type_H',
                          'reserved_room_type_G',
                          'reserved_room_type_F',
                          'reserved_room_type_E',
                          'reserved_room_type_D',
                          'reserved_room_type_C',
                          'reserved_room_type_B',
                          'reserved_room_type_A',
                          'meal_Undefined',
                          'meal_SC',
                          'meal_HB',
                          'meal_FB',
                          'meal_BB',
                          'market_segment_Undefined',
                          'market_segment_Online TA',
                          'market_segment_Offline TA/TO',
                          'market_segment_Groups',
                          'market_segment_Direct',
                          'market_segment_Corporate',
                          'market_segment_Complementary',
                          'market_segment_Aviation',
                          'hotel_Resort Hotel',
                          'hotel_City Hotel',
                          'distribution_channel_Undefined',
                          'distribution_channel_TA/TO',
                          'distribution_channel_GDS',
                          'distribution_channel_Direct',
                          'distribution_channel_Corporate',
                          'deposit_type_Refundable',
                          'deposit_type_Non Refund',
                          'deposit_type_No Deposit',
                          'customer_type_Transient-Party',
                          'customer_type_Transient',
                          'customer_type_Group',
                          'customer_type_Contract',
                          'assigned_room_type_P',
                          'assigned_room_type_L',
                          'assigned_room_type_K',
                          'assigned_room_type_I',
                          'assigned_room_type_H',
                          'assigned_room_type_G',
                          'assigned_room_type_F',
                          'assigned_room_type_E',
                          'assigned_room_type_D',
                          'assigned_room_type_C',
                          'assigned_room_type_B',
                          'assigned_room_type_A'
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

def feature_one_hot(df, features):
    dfNew = df.copy()
    for target in features:
        dfNew = feature_one_hot_one(dfNew, target)
    return dfNew

def feature_pow(df, pow, features):
    for target in features:
        name = target + '_' + str(pow)
        df[name] = df[target]**pow

def feature_multi(df, multi, features):
    for target in features:
        df[target] = df[target]*multi

def feature_standardize_multi_col(df, features, max):
    for target in features:
        df[target] = (df[target]-0)/(max-0)

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
    dfRawTrain_all = pd.read_csv(file_train)
    dfRawTrain_all = dfRawTrain_all[dfRawTrain_all.adr < 1000]
    # dfRawTrain_all = dfRawTrain_all[dfRawTrain_all.adr < 300]
    # dfRawTrain_all = dfRawTrain_all[dfRawTrain_all.adr >= 0]
    dfRawTrain_all.reset_index(drop=True, inplace=True)
    dfTrain_all = dfRawTrain_all.copy()

    dfRawTest = pd.read_csv(file_test)
    dfTest = dfRawTest.copy()

    dfIsCanceled_all = dfTrain_all['is_canceled'].to_frame() # reserve label is_canceled
    dfIsCanceled_all.reset_index(drop=True, inplace=True)
    dfAdr_all = dfTrain_all['adr'].to_frame() # reserve label adr
    dfAdr_all.reset_index(drop=True, inplace=True)
    dfIsCanceled_all = checkIsNull(dfIsCanceled_all)
    dfAdr_all = checkIsNull(dfAdr_all)
    dfAdrReal_all = ((dfIsCanceled_all['is_canceled'] == 0) * dfAdr_all['adr']).to_frame() # get real adr
    dfAdrReal_all = dfAdrReal_all.rename(columns={0: 'adr_real'})

    feature_delete(dfTrain_all, label_features)
    dfAll = pd.concat([dfTrain_all, dfTest])
    nTrain = len(dfTrain_all)
    feature_delete(dfAll, del_features)
    dfAll = checkIsNull(dfAll)
    dfAll = feature_one_hot(dfAll, one_hot_list)
    feature_delete(dfAll, del_corr_features_ohe) # delete ohe features if low correlation
    feature_standardize(dfAll, stand_features)
    feature_pow(dfAll, 2, pow_list_for_stand) # power stand features by pow
    feature_pow(dfAll, 3, pow_list_for_stand) # power stand features by pow
    # feature_delete(dfAll, del_corr_features_stand) # delete stand features if low correlation
    #multi = 2
    #feature_multi(dfAll, multi, multi_2_list_for_one_hot) # multiply one hot features by multi
    #feature_standardize_multi_col(dfAll, stand_list_for_one_hot, multi)
    dfAll = dfAll.sort_index(ascending = False, axis = 1)
    dfTrain_all = dfAll.iloc[:nTrain, :]
    dfTest = dfAll.iloc[nTrain:,  :]

    # for valid set 1
    # for valid set 2
    # for valid set 3
    # for valid set 4
    # for valid set 5
    cut = 73930 - 1
    dfTrain = dfTrain_all.iloc[:cut, :]
    dfAdr_train = dfAdr_all.iloc[:cut, :]
    dfIsCanceled_train = dfIsCanceled_all.iloc[:cut, :]
    dfAdrReal_train = dfAdrReal_all.iloc[:cut, :]
    dfRawTrain = dfRawTrain_all.iloc[:cut, :]
    # shuffle Train
    idx = np.random.RandomState(seed=5).permutation(dfTrain.index)
    # idx = np.random.permutation(dfTrain.index)
    dfTrain = dfTrain.reindex(idx)
    dfAdr_train = dfAdr_train.reindex(idx)
    dfIsCanceled_train = dfIsCanceled_train.reindex(idx)
    dfAdrReal_train = dfAdrReal_train.reindex(idx)
    dfRawTrain = dfRawTrain.reindex(idx)
    dfTrain.reset_index(drop=True, inplace=True)
    dfAdr_train.reset_index(drop=True, inplace=True)
    dfIsCanceled_train.reset_index(drop=True, inplace=True)
    dfAdrReal_train.reset_index(drop=True, inplace=True)
    dfRawTrain.reset_index(drop=True, inplace=True)

    dfValid = dfTrain_all.iloc[cut+1:nTrain, :]
    dfAdr_valid = dfAdr_all.iloc[cut+1:nTrain, :]
    dfIsCanceled_valid = dfIsCanceled_all.iloc[cut+1:nTrain, :]
    dfAdrReal_valid = dfAdrReal_all.iloc[cut+1:nTrain, :]
    dfRawValid = dfRawTrain_all.iloc[cut+1:nTrain, :]

    return dfTrain, dfRawTrain, dfAdr_train, dfIsCanceled_train, dfAdrReal_train, dfValid, dfRawValid, dfAdr_valid, dfIsCanceled_valid, dfAdrReal_valid, dfTest, dfRawTest


def correlation(file_train):
    dfTrain = pd.read_csv(file_train)

    dfIsCanceled = dfTrain['is_canceled'].to_frame() # reserve label is_canceled
    dfIsCanceled.reset_index(drop=True, inplace=True)
    dfAdr = dfTrain['adr'].to_frame() # reserve label adr
    dfAdr.reset_index(drop=True, inplace=True)
    dfIsCanceled = checkIsNull(dfIsCanceled)
    dfAdr = checkIsNull(dfAdr)
    dfAdrReal = ((dfIsCanceled['is_canceled'] == 0) * dfAdr['adr']).to_frame() # get real adr
    dfAdrReal = dfAdrReal.rename(columns={0: 'adr_real'})
    dfTrain['adr_real'] = dfAdrReal['adr_real'].to_list()

    feature_delete(dfTrain, del_features)
    dfTrain = feature_one_hot(dfTrain, one_hot_list)
    dfTrain = dfTrain.sort_index(ascending = False, axis = 1)
    dfCorr = dfTrain.corr(method ='pearson') 
    dfCorr.to_csv('./corr.csv', index=True)

# def preprocessing_revenue():
#     data_train = pd.read_csv('../data/train.csv')
#     data_train_label = pd.read_csv('../data/train_label.csv')
#     feature_delete_one(data_train_label, 'arrival_date')
#     revenue_no_quan = util.get_revenue_no_quan(data_train)
#     data_train_label.insert(len(data_train_label.columns), 'label_no_quan', revenue_no_quan, True) 
#     data_train_label.to_csv('../data/train_label_label_no_quan.csv', index=False, header=False)
