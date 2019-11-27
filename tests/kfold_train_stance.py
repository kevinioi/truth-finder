import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

import json
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from util import featureBag
from collections import defaultdict
import pickle
from sklearn import metrics

def main():
    #dictionary containing all possible features 
    featDict = featureBag.getFeatureFile("../resources/stanceFeatsV2.pickle")

    # dataSet = ([], [])
    # truthValue = None

    # #load data from all source files
    # for file_ in os.listdir("../resources//snopesData"):
    #     if file_.endswith(".json"):
    #         with open("../resources//snopesData/" + file_, 'r') as doc:
    #             fileData =  json.loads(doc.read())

    #         if fileData['Credibility'] == 'false' or fileData['Credibility'] == 'mostly false':le.HIGHEST_PROTOCOL)

    with open('../resources/pureWebSocialData.pickle', 'rb') as handle:
        dataSet = pickle.load(handle)

    dataSize = len(dataSet[0])
    wrong = [0,0]
    right = [0,0]

    for x in range(0,10):
        print(x+1)
        testStart = int(x*dataSize/10)
        testStop = int((x+1)*dataSize/10)
        one = [dataSet[0][0:testStart]+dataSet[0][testStop:], dataSet[1][0:testStart]+dataSet[1][testStop:]]
        two = [dataSet[0][testStart:testStop], dataSet[1][testStart:testStop]]
        model = traindata(one)
        tempw, tempr = testdata(two, model)
        wrong = np.add(wrong, tempw)
        right = np.add(right, tempr)
        print("**********************************")

    print(f"total correctFalse = {right[0]}")
    print(f"total correctTrue = {right[1]}")
    print(f"total wrongFalse = {wrong[0]}")
    print(f"total wrongTrue = {wrong[1]}")
    print(f"total accuracy = {(right[1]+right[0])/(wrong[0]+wrong[1]+right[1]+right[0])}")
    print(f"true accuracy = {right[1]/(right[1]+wrong[1])}")
    print(f"false accuracy = {right[0]/(right[0]+wrong[0])}")
    precisionF = (right[0]+0.0)/(right[0]+wrong[1])
    recallF = (right[0]+0.0)/(right[0]+wrong[0])
    print(f"false precision = {precisionF}")
    print(f"false recall = {recallF}")
    print(f"false F1 = {2*(precisionF*recallF)/(precisionF+recallF)}")


def traindata(dataSet):

    model = llu.train(dataSet[0], dataSet[1], '-s 0 -w1 2.7 -q')

    return model

def testdata(dataSet, model):

    p_labels, p_acc, p_vals = llu.predict(dataSet[0], dataSet[1], model, '-b 1')

    correctTrue = 0
    correctFalse = 0
    wrongTrue = 0
    wrongFalse = 0

    probs = []
    for i, x in enumerate(p_labels):
        probs.append(p_vals[i][1])
        if int(x) != int(dataSet[0][i]):
            if int(dataSet[0][i]) == 1:
                wrongTrue += 1
            else:
                wrongFalse += 1
        else:
            if int(dataSet[0][i]) == 1:
                correctTrue += 1
            else:
                correctFalse += 1
    
    #ROC/AUC calculations
    fpr, tpr, thresholds = metrics.roc_curve(dataSet[0], probs)
    auc = metrics.auc(fpr, tpr)

    print(f'AUC = {auc}')
    print(f"wrongFalse = {wrongFalse}")
    print(f"wrongTrue = {wrongTrue}")

    return [(wrongFalse, wrongTrue), (correctFalse, correctTrue)]



if __name__ == "__main__":
    main()
        