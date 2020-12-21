import math

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
    E = 0 
    for i in range(len(label)):
        E = E + (label[i] - predict[i])**2
    E = math.sqrt(E / len(label))
    return E

def zero_one_Error(label, predict):
    E = 0 
    for i in range(len(label)):
        E = E + (label[i] != predict[i])
    E = E / len(label)
    return E

def L1_Error(label, predict):
    E = 0 
    for i in range(len(label)):
        E = E + abs(label[i] - predict[i])
    E = E / len(label)
    return E

def get_iBestAlg(errors):
    for i in range(0, len(errors)):
        print("E_valid of ", i, "-th algorithm = ", errors[i], sep='')
    iBestAlg = 0
    E_min = errors[0]
    for i in range(1, len(errors)):
        if (errors[i] < E_min):
            E_min = errors[i]
            iBestAlg = i
    print("best E_valid = ", E_min, sep='')
    return iBestAlg