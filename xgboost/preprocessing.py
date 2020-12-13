import numpy as np
import pandas as pd

del_features = ['ID', 
                #'is_canceled', 
                #'adr', 
                'arrival_date_week_number', 
                'reservation_status', 
                'reservation_status_date'
                ]
stand_features = ['lead_time', 
                  'stays_in_weekend_nights', 
                  'stays_in_week_nights', 
                  'adults', 
                  'children', 
                  'babies', 
                  'previous_cancellations', 
                  'previous_bookings_not_canceled', 'booking_changes', 
                  'days_in_waiting_list', 
                  'required_car_parking_spaces', 
                  'total_of_special_requests'
                  ]

makeNaN_features = ['company',
                    'agent'
                    ]

makeOthers_list = [['country', 'PRT', 'GBR', 'other'],        # 0: feature, 1~len-2: types, len-1: label
                   #['market_segment', 'Online TA', 'Offline TA/TO', 'other'],
                   #['distribution_channel', 'TA/TO', 'Direct', 'other'],
                   ['agent', 9, -1, -2],
                   ['company', -1, 40, -2],
                   ]

one_hot_list = [['hotel', 'Resort Hotel', 'City Hotel'],        # 0: feature, 1~len-2: types, len-1: label
                ['arrival_date_year', 2015, 2016, 2017],
                ['arrival_date_month', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
                ['arrival_date_day_of_month', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                ['meal', 'Undefined', 'SC', 'BB', 'HB', 'FB'],
                ['country', 'PRT', 'GBR', 'other'],
                #['market_segment', 'Online TA', 'Offline TA/TO', 'other'],
                #['distribution_channel', 'TA/TO', 'Direct', 'other'],
                ['market_segment', 'Direct', 'Online TA', 'Offline TA/TO', 'Groups', 'Corporate', 'Complementary', 'Aviation', 'Undefined'],
                ['distribution_channel', 'TA/TO', 'Direct', 'Corporate', 'GDS', 'Undefined'],
                ['reserved_room_type', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'P'],
                ['assigned_room_type', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'P'],
                ['deposit_type', 'No Deposit', 'Non Refund', 'Refundable'],
                ['agent', 9, -1, -2],
                ['company', -1, 40, -2],
                ['customer_type', 'Contract', 'Group', 'Transient', 'Transient-party'],
                ]

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

def feature_makeNaN_one(df, feature, label):
    for i in range(len(df)):
        if (pd.isnull(df.iloc[i][feature])):
            df.iat[i, df.columns.get_loc(feature)] = label

def feature_makeNaN(df, features):
    for i in range(len(features)):
        feature_makeNaN_one(df, features[i], -1)

def feature_makeOthers_one(df, feature, types, label):
    for i in range(len(df)):
        isOther = 1
        for j in range(len(types)):
            isOther = isOther & (df.iloc[i][feature] != types[j])
            # if (isOther == 0): break
        if (isOther):
            df.iat[i, df.columns.get_loc(feature)] = label

def feature_makeOthers(df, list):
    for i in range(len(list)):
        one_list = list[i]
        feature_makeOthers_one(df, one_list[0], one_list[1:len(one_list) - 1], one_list[len(one_list) - 1])
        
def feature_one_hot_one(df, feature, types):
    names = [feature + '_' + str(a) for a in types]
    columns = [[] for _ in range(len(types))]
    for i in range(len(df)):
        for j in range(len(types)):
            columns[j].append(float(df.iloc[i][feature] == types[j]))
    
    for i in range(len(types)):
        df.insert(len(df.columns), names[i], columns[i], True) 

    feature_delete_one(df, feature)

def feature_one_hot(df, list):
    for i in range(len(list)):
        one_list = list[i]
        feature_one_hot_one(df, one_list[0], one_list[1:len(one_list)])

def feature_merge_two_multi(df, feature1, feature2, name): # feature = feature1 * feature2
    column = []
    for i in range(len(df)):
        column.append(df.iloc[i][feature1] * df.iloc[i][feature2])
    
    df.insert(len(df.columns), name, column, True) 
    feature_delete_one(df, feature1)
    feature_delete_one(df, feature2)
