import math
import numpy as np

def getFold(k, nX):
    nValid = int(nX*(1/k))
    fold = [[] for _ in range(k)]
    for i in range(k):
        if (i == k - 1):
            fold[i] = fold[i] + list(range(nValid*i, nX))
        else:
            fold[i] = fold[i] + list(range(nValid*i, nValid*(i+1)))
    return fold

def rmse(label, predict):
    label_arr = np.array(label)
    predict_arr = np.array(predict)
    E = np.sum(np.square(label_arr-predict_arr))
    E = math.sqrt(E / len(label))
    return E

def zero_one_Error(label, predict):
    label_arr = np.array(label)
    predict_arr = np.array(predict)
    E = np.sum(np.not_equal(label_arr, predict_arr))
    E = E / len(label)
    return E

def L1_Error(label, predict):
    label_arr = np.array(label)
    predict_arr = np.array(predict)
    E = np.sum(np.absolute(label_arr - predict_arr))
    E = E / len(label)
    return E

def get_iBestAlg(errors, param, param_value):
    for i in range(0, len(param_value)):
        print(i, "-th algorithm:", sep='')
        for j in range(0, len(param)):
            print(" ", param[j], "=", param_value[i][j], sep='')
    for i in range(0, len(errors)):
        print("E_train_out of ", i, "-th algorithm = ", errors[i], sep='')
    iBestAlg = 0
    E_min = errors[0]
    for i in range(1, len(errors)):
        if (errors[i] < E_min):
            E_min = errors[i]
            iBestAlg = i
    print("best E_train_out = ", E_min, sep='')
    return iBestAlg