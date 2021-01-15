import numpy as np
import pandas as pd

def encode_target(dfRaw, targetCol, dfEncode, encodeCol): # Add column to df with integers for the target.
    targets = dfRaw[targetCol].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    dfEncode[encodeCol] = dfRaw[targetCol].replace(map_to_int)
    return (dfEncode, targets)

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
