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
from multiprocessing import Pool
from util import textProcessor
import numpy as np
from sklearn import metrics

"""
    Generates the final data needed for Joing model (CRF) training/testing
    Also trains and tests the model

    **BEFORE USING THIS**

    ensure: 
    - features/stances of articles is up to date--->    ../resources//contentTrain//out
    - linguistic features are up to date --->     ../resources//dataFiles//lingFeatures//out/
    - reliability file is upto date -----> ../resources//compiledReliabilityDict702.txt
"""

def makeJointData():
    # [[truthvalues], [{features}], [file_name]]
    fullDataSet = [[], [], []]
    
    #reliability data
    with open("../resources//compiledReliabilityDict707.txt", "r") as relFile:
        relDict = json.load(relFile)
    relDict = defaultdict(lambda: -1, relDict)


    for file_ in os.listdir("../resources//contentTrain//out"):
        if file_.endswith(".json"):

            #features = 1:reliability, 2:prob support?, 3: prob refute?, 4+: lingFeats
            # fullClaimData = [truth value,{features},file_]
            fullClaimData = [-1,defaultdict(lambda: 0),""]

            with open("../resources//contentTrain//out/" + file_, 'r') as doc:
                claimData =  json.loads(doc.read())
            
            if len(claimData[0]) > 0:
                fullClaimData[0]= (claimData[0])[0]#truth value
                fullClaimData[2] = file_

                for article in claimData[1]:
                    articleFeats = {}
                    sumprobabilities = [0,0]

                    #avg probabilities of first 5 snippets in article
                    for i, snippet in enumerate(article[0][:5]):
                        sumprobabilities[0] += snippet[0][0]
                        sumprobabilities[1] += snippet[0][1]
                    sumprobabilities[0] /= i+1
                    sumprobabilities[1] /= i+1
                    
                    #get ling feats
                    with open("../resources//dataFiles//lingFeatures//out/"+file_ , 'r') as lFile:
                        lFileDict = json.load(lFile)

                    articleFeats[2] = sumprobabilities[0]#prob refute
                    articleFeats[3] = sumprobabilities[1]#prob support
                    articleFeats.update(lFileDict[article[2]])#Use url as key in lingfeats
                    for feat in articleFeats:
                        fullClaimData[1][int(feat)] += articleFeats[feat] #add features to dictionary representing aggregate claim features

                    #get reliability
                    reliability = relDict[article[1]]
                    if reliability == -1:#new domian
                        reliability = 0.5
                    elif reliability < 0.25:#baseline reliability
                        reliability = 0.25
                    fullClaimData[1][1] += reliability

                fullDataSet[0].append(fullClaimData[0])#truth value
                fullDataSet[1].append(dict(fullClaimData[1])) #feature dict
                fullDataSet[2].append(fullClaimData[2])#file name

    
    with open('../resources//jointModelDataV3.pickle', 'wb') as handle:
        pickle.dump(fullDataSet, handle, protocol=pickle.HIGHEST_PROTOCOL)


def trainJoint(dataSet):

    # newData = [[], []]

    # for i, claim in enumerate(dataSet[1]):
    #     newData[0].append(dataSet[0][i])
    #     newDict = {}
    #     newDict[1] = claim[1]
    #     newDict[2] = claim[2]
    #     newDict[3] = claim[3]
    #     newData[1].append(newDict)


    model = llu.train(dataSet[0], dataSet[1], '-s 6 -w1 3.1 -q') # -v 10

    # llu.save_model("../resources//models/jointModelSample.model",model)
    
    return model


def testJoint(dataSet, model):
    # model = llu.load_model("../resources//models/jointModelSample.model")

    # newData = [[], []]

    # for i, claim in enumerate(dataSet[1]):
    #     newData[0].append(dataSet[0][i])
    #     newDict = {}
    #     newDict[1] = claim[1]
    #     newDict[2] = claim[2]
    #     newDict[3] = claim[3]
    #     newData[1].append(newDict)

    p_labels, p_acc, p_vals = llu.predict(dataSet[0], dataSet[1], model)

    correctTrue = 0
    correctFalse = 0
    wrongTrue = 0
    wrongFalse = 0
    for i, x in enumerate(p_labels):
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
    print(f"wrongFalse = {wrongFalse}")
    print(f"wrongTrue = {wrongTrue}")

    return [(wrongFalse, wrongTrue), (correctFalse, correctTrue)]


if __name__ == "__main__":
    # makeJointData()

    with open('../resources//jointModelDataV3.pickle', 'rb') as handle:
        dataSet = pickle.load(handle)

    normDict = defaultdict(lambda : 0)
    for i, claim in enumerate(dataSet[1]):
        for feature in claim:
            if claim[feature] > normDict[feature]: 
                normDict[feature] = claim[feature] 

    for i, claim in enumerate(dataSet[1]):
        for feature in claim:
            dataSet[1][i][feature] /= normDict[feature]

    dataSize = len(dataSet[0])

    wrong = [0,0]
    right = [0,0]
    roc = []

    for x in range(0,10):
        print(x+1)
        testStart = int(x*dataSize/10)
        testStop = int((x+1)*dataSize/10)
        one = [dataSet[0][0:testStart]+dataSet[0][testStop:], dataSet[1][0:testStart]+dataSet[1][testStop:]]
        two = [dataSet[0][testStart:testStop], dataSet[1][testStart:testStop]]
        model = trainJoint(one)
        tempw, tempr = testJoint(two, model)
        wrong = np.add(wrong, tempw)
        right = np.add(right, tempr)


        #true positive rate, false positive rate
        roc.append((right[1]/(right[1]+wrong[1]), wrong[0]/(wrong[0]+right[0])))

        print("**********************************")

    roc.sort(key=lambda x: x[0])

    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
    tpr = []
    fpr = []
    for x in roc:
        tpr.append(x[0])
        fpr.append(x[1])

    auc = metrics.auc(tpr, fpr)
    print(f'AUC: {auc}')

    # print(roc)
    # auc = 0

    # for i, point in enumerate(roc):
    #     if i > 0:
    #         auc += (point[0] - roc[i-1][0])*roc[i-1][1]
    #         auc +=  (point[0] - roc[i-1][0])*point[1]/2

    # print(auc)

    print(f"total correctFalse = {right[0]}")
    print(f"total correctTrue = {right[1]}")
    print(f"total wrongFalse = {wrong[0]}")
    print(f"total wrongTrue = {wrong[1]}")

    print(f"total accuracy = {(right[1]+right[0])/(wrong[0]+wrong[1]+right[1]+right[0])}")
    print(f"true accuracy {right[1]/(right[1]+wrong[1])}")
    print(f"false accuracy {right[0]/(right[0]+wrong[0])}")