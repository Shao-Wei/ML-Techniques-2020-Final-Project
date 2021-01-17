import numpy as np
import pandas as pd

def ohe_target(df, targetCol):
    pOHE = pd.get_dummies(df[targetCol], prefix = targetCol)
    df.drop(targetCol, inplace=True, axis = 1)
    dfNew = pd.concat([df, pOHE], axis = 1)
    return dfNew

def checkIsNull(df):
    if df.isnull().values.any():
        features = list(df.columns[:]) # get list of all features
        # print("* features:", features, sep="\n", end='\n\n')
        for target in features:
            if df[target].isnull().values.any():
                print("    IsNULL column filled up: ", target, end='\n')
                df[target] = df[target].replace(np.nan, 0)
    return df

## feature classes
label_features = ['reservation_status', 'reservation_status_date', 'is_canceled', 'adr']
del_features = ['ID', 'arrival_date_year', 'arrival_date_week_number', 'country', 'agent', 'company']
one_hot_list = ['hotel', 'arrival_date_month', 'arrival_date_day_of_month', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type']
# corr < 0.05
del_corr_features_ohe = [ 'reserved_room_type_P', 'reserved_room_type_L', 'reserved_room_type_B', 'meal_Undefined', 'meal_SC', 'meal_FB', 'market_segment_Undefined', 'market_segment_Aviation', 'distribution_channel_Undefined', 'distribution_channel_GDS', 'deposit_type_Refundable', 'customer_type_Group', 'customer_type_Contract', 'assigned_room_type_P', 'assigned_room_type_L', 'assigned_room_type_K', 'assigned_room_type_B']
stand_features = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list', 'required_car_parking_spaces', 'total_of_special_requests' ]
pow_list_for_stand = ['total_of_special_requests', 'required_car_parking_spaces', 'lead_time', 'children' ]

# modified from mstf7249
# rm features w/ low correlation, add pow 2 3 for important features
def preprocessing_enhanced(file_train, file_test):
    # get raw data
    dfRawTrain_all = pd.read_csv(file_train)
    dfRawTrain_all = dfRawTrain_all[dfRawTrain_all.adr < 1000]
    dfRawTrain_all.reset_index(drop=True, inplace=True)
    dfTrain_all = dfRawTrain_all.copy()
    dfRawTest = pd.read_csv(file_test)
    dfTest = dfRawTest.copy()

    # reserve label
    dfIsCanceled_all = dfTrain_all['is_canceled'].to_frame()
    dfIsCanceled_all.reset_index(drop=True, inplace=True)
    dfIsCanceled_all = checkIsNull(dfIsCanceled_all)
    dfAdr_all = dfTrain_all['adr'].to_frame()
    dfAdr_all.reset_index(drop=True, inplace=True)
    dfAdr_all = checkIsNull(dfAdr_all)
    dfAdrReal_all = ((dfIsCanceled_all['is_canceled'] == 0) * dfAdr_all['adr']).to_frame() # get real adr
    dfAdrReal_all = dfAdrReal_all.rename(columns={0: 'adr_real'})

    # concat train & test
    for f in label_features:
        del dfTrain_all[f]
    dfAll = pd.concat([dfTrain_all, dfTest])
    dfAll = checkIsNull(dfAll)
    nTrain = len(dfTrain_all)

    # drop and enhance
    for f in del_features:
        del dfAll[f]
    for f in one_hot_list:
        dfAll = ohe_target(dfAll, f)
    for f in del_corr_features_ohe:
        del dfAll[f]
    for f in stand_features:
        dfAll[f] = (dfAll[f] - dfAll[f].min()) / (dfAll[f].max() - dfAll[f].min())
    for f in pow_list_for_stand:
        name2 = f + '_' + str(2)
        name3 = f + '_' + str(3)
        dfAll[name2] = dfAll[f]**2
        dfAll[name3] = dfAll[f]**3

    dfAll = dfAll.sort_index(ascending = False, axis = 1)
    dfTrain_all = dfAll.iloc[:nTrain, :]
    dfTest = dfAll.iloc[nTrain:,  :]

    # divide into train and valid
    cut = 73930 - 1
    dfTrain = dfTrain_all.iloc[:cut, :]
    dfAdr_train = dfAdr_all.iloc[:cut, :]
    dfIsCanceled_train = dfIsCanceled_all.iloc[:cut, :]
    dfAdrReal_train = dfAdrReal_all.iloc[:cut, :]
    dfRawTrain = dfRawTrain_all.iloc[:cut, :]
    
    # idx = np.random.RandomState(seed=5).permutation(dfTrain.index)
    # dfTrain = dfTrain.reindex(idx)
    # dfAdr_train = dfAdr_train.reindex(idx)
    # dfIsCanceled_train = dfIsCanceled_train.reindex(idx)
    # dfAdrReal_train = dfAdrReal_train.reindex(idx)
    # dfRawTrain = dfRawTrain.reindex(idx)
    # dfTrain.reset_index(drop=True, inplace=True)
    # dfAdr_train.reset_index(drop=True, inplace=True)
    # dfIsCanceled_train.reset_index(drop=True, inplace=True)
    # dfAdrReal_train.reset_index(drop=True, inplace=True)
    # dfRawTrain.reset_index(drop=True, inplace=True)

    dfValid = dfTrain_all.iloc[cut+1:nTrain, :]
    dfAdr_valid = dfAdr_all.iloc[cut+1:nTrain, :]
    dfIsCanceled_valid = dfIsCanceled_all.iloc[cut+1:nTrain, :]
    dfAdrReal_valid = dfAdrReal_all.iloc[cut+1:nTrain, :]
    dfRawValid = dfRawTrain_all.iloc[cut+1:nTrain, :]

    return dfTrain, dfRawTrain, dfAdr_train, dfIsCanceled_train, dfAdrReal_train, dfValid, dfRawValid, dfAdr_valid, dfIsCanceled_valid, dfAdrReal_valid, dfTest, dfRawTest


## feature classes
oheList = ['hotel', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type']
scaleList = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list', 'required_car_parking_spaces', 'total_of_special_requests']
donothingList = ['is_repeated_guest']
dropList = ['ID', 'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month', 'country', 'agent', 'company'] # entries of no use
labelList = ['is_canceled', 'adr', 'reservation_status', 'reservation_status_date']

## preprocess function
# dfRawTrain used in deriving revenue 
# taking test csv into consideration - training & testing have large difference on the distribution
def preprocess(trainFileName, testFileName):
    print("  Preprocessing..", trainFileName)
    dfRawTrain = pd.read_csv(trainFileName, header = 0, sep=',') # read in csv
    dfRawTrain_dup = dfRawTrain.copy()
    dfRawTest = pd.read_csv(testFileName, header = 0, sep=',') # read in csv
    dfRawTest_dup = dfRawTest.copy()

    dfIsCanceled = dfRawTrain[['is_canceled']] # reserve label is_canceled
    dfAdr = dfRawTrain[['adr']] # reserve label adr
    
    dfAll = pd.concat([dfRawTrain, dfRawTest])
    nTrain = len(dfRawTrain)

    for target in dropList:
        dfAll.drop(target, inplace=True, axis=1)
    for target in labelList:
        dfAll.drop(target, inplace=True, axis=1)
    for target in oheList:
        dfAll = ohe_target(dfAll, target)
    for target in scaleList:
        dfAll[target] = (dfAll[target] - dfAll[target].mean()) / dfAll[target].std()

    dfAll = checkIsNull(dfAll)
    dfAdr = checkIsNull(dfAdr)
    dfIsCanceled = checkIsNull(dfIsCanceled)
    dfAdrReal = ((dfIsCanceled['is_canceled'] == 0) * dfAdr['adr']).to_frame() # get real adr
    dfAdrReal = dfAdrReal.rename(columns={0: 'adr_real'})
    
    dfAll = dfAll.sort_index(ascending = False, axis = 1)
    dfEncodedTrain = dfAll.iloc[:nTrain, :]
    dfEncodedTest = dfAll.iloc[nTrain:,  :]

    return dfEncodedTrain, dfAdr, dfIsCanceled, dfAdrReal, dfRawTrain_dup, dfEncodedTest, dfRawTest_dup

def preprocess_train(trainFileName):
    print("  Preprocessing..", trainFileName)
    dfRawTrain = pd.read_csv(trainFileName, header = 0, sep=',') # read in csv
    dfRawTrain_dup = dfRawTrain.copy()
    dfRawTrain.dropna() # drop data w/ missing entries

    dfIsCanceled = dfRawTrain[['is_canceled']] # reserve label is_canceled
    dfAdr = dfRawTrain[['adr']] # reserve label adr

    for target in dropList:
        dfRawTrain.drop(target, inplace=True, axis=1)
    for target in labelList:
        dfRawTrain.drop(target, inplace=True, axis=1)
    for target in oheList:
        dfRawTrain = ohe_target(dfRawTrain, target)
    for target in scaleList:
        dfRawTrain[target] = (dfRawTrain[target] - dfRawTrain[target].mean()) / dfRawTrain[target].std()

    dfRawTrain = checkIsNull(dfRawTrain)
    dfAdr = checkIsNull(dfAdr)
    dfIsCanceled = checkIsNull(dfIsCanceled)
    dfAdrReal = ((dfIsCanceled['is_canceled'] == 0) * dfAdr['adr']).to_frame() # get real adr
    dfAdrReal = dfAdrReal.rename(columns={0: 'adr_real'})
    
    feature_train = list(dfRawTrain.columns[:])
    # print("* dfRawTrain.head()", dfRawTrain.head(), sep='\n', end='\n\n')
    dfEncodedTrain = dfRawTrain.sort_index(ascending = False, axis = 1) 

    return dfEncodedTrain, dfAdr, dfIsCanceled, dfAdrReal, dfRawTrain_dup, feature_train

def preprocess_test(testFileName, feature_train):
    print("  Preprocessing..", testFileName)
    dfRawTest = pd.read_csv(testFileName, header = 0, sep=',') # read in csv
    dfRawTest_dup = dfRawTest.copy()
    # dfRawTest.dropna() # drop data w/ missing entries

    for target in dropList:
        dfRawTest.drop(target, inplace=True, axis=1)
    # for target in labelList:
    #     dfRawTest.drop(target, inplace=True, axis=1)
    for target in oheList:
        dfRawTest = ohe_target(dfRawTest, target)
    for target in scaleList:
        dfRawTest[target] = (dfRawTest[target] - dfRawTest[target].mean()) / dfRawTest[target].std()

    dfRawTest = checkIsNull(dfRawTest)
    feature_test = list(dfRawTest.columns[:])
    feature_missing = [i for i in feature_train if i not in feature_test] # append missing feature columns
    print("    missing features: ", feature_missing, sep = ' ', end = '\n\n')
    for target in feature_missing:
        dfRawTest[target] = '0'
    # print("* dfRawTest.head()", dfRawTest.head(), sep='\n', end='\n\n')
    dfEncodedTest = dfRawTest.sort_index(ascending = False, axis = 1)

    return dfEncodedTest, dfRawTest_dup
