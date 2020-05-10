'''
    Module used to train the distance supervision based credibility classifier
'''

import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from util import featureBag
from util import textProcessor
from collections import defaultdict

import argparse
import numpy as np
import pickle
import json

def main(args):
    training(CRED_DATASET, save=args.modelName)


def training(dataSet, save = False):
    """
        trains the L1 logistic regresssion model

        return: trained liblinear model object

        param: dataSet: training data in form:
                    [[truthVales], [{features}]]
    """

    ratio = 1/(sum(dataSet[0])/len(dataSet[0]))
    model = llu.train(dataSet[0], dataSet[1], f'-s 6 -w1 {ratio} -q')

    if save:
        savePath = os.path.join("resources","models",f"{save}.model")
        llu.save_model(savePath,model)
    return model

def testModel(dataSet, reliabilityDictPath, model):
    """
    return: list of two tuples containing the summary data of the testing
        [(wrongFalse, wrongTrue), (correctFalse, correctTrue)]

    param: dataSet: list containing [[truthvalues], [{features}]], truth values list can be empty
    param: model: trained/loaded liblinear model object to be used in the prediction

    """
    
    p_labels, p_acc, p_vals = llu.predict(dataSet[0], dataSet[1], model, '-b 1')

    # Load reliability
    # default dict, default -1
    with open(reliabilityDictPath, "r") as relFile:
        relDict = json.load(relFile)
    relDict = defaultdict(lambda: -1, relDict)

    judgements = []
    current_claim = ""
    current_claim_truth = -1

    rocCalc = [[],[]]
    sumCurrentProbs = [0,0]
    sCount = 0
    articleCount = 0
    for i, probs in enumerate(p_vals):#run through results adjusting for reliability and summarizing for full claims
        if (i>0) and (current_claim != dataSet[3][i]):#not first claim in dataset AND new claim
            if articleCount <= maxArticleCount:    
                if sumCurrentProbs[0] <= sumCurrentProbs[1]:
                    judgements.append((1, current_claim_truth))
                else:
                    judgements.append((0,current_claim_truth))

                rocCalc[0].append(current_claim_truth)
                rocCalc[1].append(sumCurrentProbs[1]/sCount)
            articleCount=0
            sCount = 0
            sumCurrentProbs = [0,0]
            
        articleCount +=1
        #sum probabilities for the current claim
        sCount += 1
        current_claim_truth = dataSet[0][i]
        current_claim = dataSet[3][i]
        reliability = relDict[dataSet[2][i]]

        if reliability == -1:
            reliability = 0.5
        elif reliability < 0.1:
            reliability = 0.1
        else:
            reliability = reliability*2/3
        sumCurrentProbs[0] += probs[0]*reliability
        sumCurrentProbs[1] += probs[1]*reliability

    correctTrue = 0
    correctFalse = 0
    wrongTrue = 0
    wrongFalse = 0

    for i, x in enumerate(judgements):
        if x[0] != x[1]:
            if x[1] == 1:
                wrongTrue += 1
            else:
                wrongFalse +=1
        else:
            if x[1] == 1:
                correctTrue += 1
            else:
                correctFalse += 1

    #ROC/AUC calculations
    fpr, tpr, thresholds = metrics.roc_curve(rocCalc[0], rocCalc[1])
    auc = metrics.auc(fpr, tpr)

    print(f'AUC = {auc}')
    print(f"claims = {i+1}")
    print(f"wrongFalse = {wrongFalse}")
    print(f"wrongTrue = {wrongTrue}")
    print(f'accuracy = {1-(wrongTrue+wrongFalse)/(i+1)}')

    return [(wrongFalse, wrongTrue), (correctFalse, correctTrue), auc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input parameters')
    parser.add_argument('--modelName', metavar='--modelName', type=str,
                help='identifier of the model being trained')
    args = parser.parse_args()

    main(args)