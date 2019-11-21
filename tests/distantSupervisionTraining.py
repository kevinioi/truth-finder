import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

import json
from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from util import featureBag
import pickle
from collections import defaultdict 


def testModel():
    
    #load dataset
    with open('../resources//properDistantSupervisionDataV1.pickle', 'rb') as handle:
        dataSet = pickle.load(handle)

    #load trained model
    model = llu.load_model("../resources//models/distantSupervisionSample.model")

    #run predictions
    p_labels, p_acc, p_vals = llu.predict(dataSet[0][:1881], dataSet[1][:1881], model, '-b 1')


    # Load reliability
    # default dict, default -1
    with open("../resources//compiledReliabilityDict502.txt", "r") as relFile:
        relDict = json.load(relFile)
    relDict = defaultdict(lambda: -1    , relDict)

    judgements = []
    current_claim = ""
    current_claim_truth = -1
    sumCurrentProbs = [0,0]
    for i, probs in enumerate(p_vals):

        if (i>0) and (current_claim != dataSet[3][i]):#not first claim in dataset AND new claim
            if sumCurrentProbs[0] < sumCurrentProbs[1]:
                judgements.append((1, current_claim_truth))
            else:
                judgements.append((0,current_claim_truth))
            sumCurrentProbs = [0,0]
        
        current_claim_truth = dataSet[0][i]
        current_claim = dataSet[3][i]
        reliability = relDict[dataSet[2][i]]
        if reliability == -1:
            reliability = 0.5
        elif reliability < 0.1:
            reliability = 0.1
        sumCurrentProbs[0] += probs[0]*reliability
        sumCurrentProbs[1] += probs[1]*reliability

    correct = 0
    wrong = 0
    gotFalsewrong = 0
    gotTruewrong = 0

    for x in judgements:
        if x[0] == x[1]:
            correct += 1
        else:
            wrong += 1
            if x[1] == 1:
                gotTruewrong += 1
            else:
                gotFalsewrong += 1

    print(f"correct: {correct}")
    print(f"wrong: {wrong}")
    print(f"False idenfified as true: {gotFalsewrong}")
    print(f"True identified as false: {gotTruewrong}")



def training():
    with open('../resources//properDistantSupervisionDataV2.pickle', 'rb') as handle:
        dataSet = pickle.load(handle)

    model = llu.train(dataSet[0][1881:], dataSet[1][1881:], '-s 6 -w1 2.7') # -v 10
    llu.save_model("../resources//models/distantSupervisionV2M1.model",model)


if __name__ == "__main__":
    # training()
    testModel()