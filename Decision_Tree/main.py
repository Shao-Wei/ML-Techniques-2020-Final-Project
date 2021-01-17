import numpy as np
import pandas as pd
import preprocessing as pp
import training as tr
import util
import sys

## options argv[1]
# 0 combined
# 1 seperated

algs_all = [[[18, 20, 1000]]]

## path declaration
pTraincsv = "../data/train.csv"
pTrainLabel = '../data/train_label.csv'
pTestcsv = "../data/test.csv"
pTestNoLabel = '../data/test_nolabel.csv'
pTestLabel = '../data/test_label.csv'


if sys.argv[1] == '0':
    print("[Info] Start revenue prediction: combined")
else:
    print("[Info] Start revenue prediction: seperated")

## get data, alias
print("Start preprocessing ...")
dfTrain, dfRawTrain, dfAdr_train, dfIsCanceled_train, dfAdrReal_train, dfValid, dfRawValid, dfAdr_valid, dfIsCanceled_valid, dfAdrReal_valid, dfTest, dfRawTest = pp.preprocessing_enhanced(pTraincsv, pTestcsv) 
# dfTrain, dfAdr, dfIsCanceled, dfAdrReal, dfRawTrain, dfTest, dfRawTest = pp.preprocess(pTraincsv, pTestcsv) 
# dfTrain, dfAdr, dfIsCanceled, dfAdrReal, dfRawTrain, feature_train = pp.preprocess_train(pTraincsv)
# dfTest, dfRawTest = pp.preprocess_test(pTestcsv, feature_train)
print("Preprocessing done ...")
for count in range(len(algs_all)):
    algs = algs_all[count]
    fLog = open(sys.argv[2], "a")
    print("    Training ..", algs[0])
    print("Start training ..")
    if sys.argv[1] == '0':
        modelList_adr_real, bestSetting, E_min = tr.train_adr(dfTrain, dfAdrReal_train, algs, 5)
    else:
        modelList_adr, bestSetting_adr, E_min_adr = tr.train_adr(dfTrain, dfAdr_train, algs, 5)
        modelList_is_canceled, bestSetting_isCanceled, E_min_isCanceled = tr.train_adr(dfTrain, dfIsCanceled_train, algs, 5)
    print("Training done ...")

    if sys.argv[1] == '0':
        # fLog.write("," + "[" + str(bestSetting[0]) + "," + str(bestSetting[1]) + "," + str(bestSetting[2]) + "]")
        fLog.write("," + str(E_min))
        adr_real_train = util.getAdrEnsemblePredict(modelList_adr_real, dfTrain)
        Ein_adr_real = util.rmse(dfAdrReal_train['adr_real'].tolist(), adr_real_train)
        print("Ein_adr_real: ", Ein_adr_real)
        fLog.write("," + str(Ein_adr_real))
        adr_real_valid = util.getAdrEnsemblePredict(modelList_adr_real, dfValid)
        Eval_adr_real = util.rmse(dfAdrReal_valid['adr_real'].tolist(), adr_real_valid)
        print("Eval_adr_real: ", Eval_adr_real)
        fLog.write("," + str(Eval_adr_real))
        fLog.write(", -, -, -, -")
    else:
        # fLog.write("," + "[" + str(bestSetting_adr[0]) + "," + str(bestSetting_adr[1]) + "," + str(bestSetting_adr[2]) + "]")
        fLog.write("," + str(E_min_adr))
        adr_train = util.getAdrEnsemblePredict(modelList_adr, dfTrain)
        Ein_adr = util.rmse(dfAdr_train['adr'].tolist(), adr_train)
        print("Ein_adr: ", Ein_adr)
        fLog.write("," + str(Ein_adr))
        adr_valid = util.getAdrEnsemblePredict(modelList_adr, dfValid)
        Eval_adr = util.rmse(dfAdr_valid['adr'].tolist(), adr_valid)
        print("Eval_adr: ", Eval_adr)
        fLog.write("," + str(Eval_adr))
        # fLog.write("," + "[" + str(bestSetting_isCanceled[0]) + "," + str(bestSetting_isCanceled[1]) + "," + str(bestSetting_isCanceled[2]) + "]")
        fLog.write("," + str(E_min_isCanceled))
        isCanceled_train = util.getIsCanceledEnsemblePredict(modelList_is_canceled, dfTrain)
        Ein_isCanceled = util.L1_Error(np.ravel(dfIsCanceled_train['is_canceled']), isCanceled_train)
        print("Ein_isCanceled: ", Ein_isCanceled)
        fLog.write("," + str(Ein_isCanceled))
        isCanceled_valid = util.getIsCanceledEnsemblePredict(modelList_is_canceled, dfValid)
        Eval_isCanceled = util.L1_Error(np.ravel(dfIsCanceled_valid['is_canceled']), isCanceled_valid)
        print("Eval_isCanceled: ", Eval_isCanceled)
        fLog.write("," + str(Eval_isCanceled))

    print("Computing revenue ...")
    if sys.argv[1] == '0':
        revenue_train = util.predict_revenue_ensemble_adr_isCanceled_combined(dfTrain, dfRawTrain, modelList_adr_real, 0)
        revenue_valid = util.predict_revenue_ensemble_adr_isCanceled_combined(dfValid, dfRawValid, modelList_adr_real, 1)
        revenue_test = util.predict_revenue_ensemble_adr_isCanceled_combined(dfTest, dfRawTest, modelList_adr_real, 2)
    else:
        revenue_train = util.predict_revenue_ensemble_adr_isCanceled_separated(dfTrain, dfRawTrain, modelList_adr, modelList_is_canceled, 0)
        revenue_valid = util.predict_revenue_ensemble_adr_isCanceled_separated(dfValid, dfRawValid, modelList_adr, modelList_is_canceled, 1)
        revenue_test = util.predict_revenue_ensemble_adr_isCanceled_separated(dfTest, dfRawTest, modelList_adr, modelList_is_canceled, 2)
    print("Computing revenue done ...")

    # train quantization
    dfTrain_label = pd.read_csv(pTrainLabel)
    train_label = dfTrain_label['label'].tolist()
    revenue_all = (np.array(revenue_train) + np.array(revenue_valid)).tolist()
    revenue_train = revenue_all[:511]
    revenue_valid = revenue_all[511:]
    valid_label = train_label[511:]
    train_label = train_label[:511]

    Eval_L1 = util.L1_Error(valid_label, util.revenue_quantization(revenue_valid))
    print("Eval_revenue (un-trained) = ", Eval_L1)
    fLog.write("," + str(Eval_L1))

    revenue_train, revenue_valid, revenue_test = tr.train_quantize(revenue_train, train_label, revenue_valid, valid_label, revenue_test, algs)
    Eval_L1 = util.L1_Error(valid_label, revenue_valid)
    print("Eval_revenue (trained) = ", Eval_L1)
    fLog.write("," + str(Eval_L1))

    fLog.write("\n")
    fLog.close()
    # # write predict
    # dfTest_nolabel = pd.read_csv(pTestNoLabel)
    # dfTest_nolabel['label'] = revenue_test
    # dfTest_nolabel.to_csv(pTestLabel, index=False)