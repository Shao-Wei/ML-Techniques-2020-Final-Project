month_dic = {
    'January': 0,
    'February': 1,
    'March': 2,
    'April': 3,
    'May': 4, 
    'June': 5, 
    'July': 6, 
    'August': 7, 
    'September': 8, 
    'October': 9, 
    'November': 10, 
    'December': 11
}

def squaredError(label, predict):
    E = 0 
    for i in range(len(label)):
        E = E + (label[i] - predict[i])**2
    E = E / len(label)
    return E

def L1_Error(label, predict):
    E = 0 
    for i in range(len(label)):
        E = E + abs(label[i] - predict[i])
    E = E / len(label)
    return E

def get_bestAlg(algs, errors):
    for i in range(0, len(algs)):
        print("E_valid of ", i, "-th algorithm = ", errors[i], sep='')
    bestAlg = algs[0]
    E_min = errors[0]
    for i in range(1, len(algs)):
        if (errors[i] < E_min):
            E_min = errors[i]
            bestAlg = algs[i]
    print("best E_valid = ", E_min, sep='')
    return bestAlg

def delete_non_date(list):
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

def get_revenue_predict(df, adr_preds):
    revenue = [[[] for _ in range(12)] for _ in range(3)]
    for i in range(3):
        for j in range(12):
            revenue[i][j] = [0]*31
    for i in range(len(df)):
        revenue[df.iloc[i]['arrival_date_year']-2015][month_dic[df.iloc[i]['arrival_date_month']]][df.iloc[i]['arrival_date_day_of_month']-1] += (df.iloc[i]['stays_in_weekend_nights']+df.iloc[i]['stays_in_week_nights'])*adr_preds[i]
    revenue = delete_non_date(revenue)
    return [c for a in revenue for b in a for c in b]

def get_revenue_no_quan(df):
    revenue = [[[] for _ in range(12)] for _ in range(3)]
    for i in range(3):
        for j in range(12):
            revenue[i][j] = [0]*31
    for i in range(len(df)):
        if (df.iloc[i]['is_canceled']):
            revenue[df.iloc[i]['arrival_date_year']-2015][month_dic[df.iloc[i]['arrival_date_month']]][df.iloc[i]['arrival_date_day_of_month']-1] += 0
        else:
            revenue[df.iloc[i]['arrival_date_year']-2015][month_dic[df.iloc[i]['arrival_date_month']]][df.iloc[i]['arrival_date_day_of_month']-1] += (df.iloc[i]['stays_in_weekend_nights']+df.iloc[i]['stays_in_week_nights'])*df.iloc[i]['adr']
    revenue = delete_non_date(revenue)
    return [c for a in revenue for b in a for c in b]