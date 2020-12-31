import math

## aux for get_revenue_train
dicMonth = {
    'January': 0, 'February': 1, 'March': 2,
    'April': 3, 'May': 4, 'June': 5, 
    'July': 6, 'August': 7, 'September': 8, 
    'October': 9, 'November': 10, 'December': 11
}

def remove_extra_date_train(list):
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

def remove_extra_date_test(list):
    list_out = list
    list_out[0][30:31] = []
    list_out[2][30:31] = []
    return list_out

## error matrics
def rmse(label, predict):
    E = 0 
    for i in range(len(label)):
        E = E + (label[i] - predict[i])**2
    E = math.sqrt(E / len(label))
    return E

def L1_Error(label, predict):
    E = 0 
    for i in range(len(label)):
        E = E + abs(label[i] - predict[i])
    E = E / len(label)
    return E

## util
def get_bestAlg(errors):
    for i in range(0, len(errors)):
        print("  E_valid of ", i, "-th algorithm = ", errors[i], sep='')
    bestAlg = 0
    E_min = errors[0]
    for i in range(1, len(errors)):
        if (errors[i] < E_min):
            E_min = errors[i]
            bestAlg = i
    print("  best E_valid = ", E_min, sep='')
    return bestAlg

def get_revenue_train(df, adr_preds):
    revenue = [[[] for _ in range(12)] for _ in range(3)]
    for i in range(3):
        for j in range(12):
            revenue[i][j] = [0]*31
    for i in range(len(df)):
        revenue[df.iloc[i]['arrival_date_year']-2015][dicMonth[df.iloc[i]['arrival_date_month']]][df.iloc[i]['arrival_date_day_of_month']-1] += (df.iloc[i]['stays_in_weekend_nights']+df.iloc[i]['stays_in_week_nights'])*adr_preds[i]
    revenue = remove_extra_date_train(revenue) # rm dates which do not exist in train set 
    return [c for a in revenue for b in a for c in b]

def get_revenue_test(df, adr_preds):
    revenue = [[] for _ in range(5)]
    for i in range(5):
        revenue[i] = [0]*31
    for i in range(len(df)):
        revenue[dicMonth[df.iloc[i]['arrival_date_month']]-3][df.iloc[i]['arrival_date_day_of_month']-1] += (df.iloc[i]['stays_in_weekend_nights']+df.iloc[i]['stays_in_week_nights'])*adr_preds[i]
    revenue = remove_extra_date_test(revenue) # rm dates which do not exist in test set
    return [b for a in revenue for b in a]

def revenue_quantization(revenue):
    revenue_quan = [int(math.floor(x/10000)) for x in revenue]
    revenue_quan = [0 if x < 0 else x for x in revenue_quan] # set negative values to zero
    return revenue_quan

def predict_revenue_ensemble_adr_isCanceled_combined(df, dfRaw, modelList, fMode): # fMode = 1 for train, = 0 for test
    preds = [0] * df.shape[0]
    for i in range(len(modelList)):
        preds = preds + modelList[i].predict(df)
    preds = preds / len(modelList)

    if (fMode):
        revenue = get_revenue_train(dfRaw, preds)
    else:
        revenue = get_revenue_test(dfRaw, preds)
    revenue = revenue_quantization(revenue)
    return revenue