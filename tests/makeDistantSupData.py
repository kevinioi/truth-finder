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

"""
    Generates the final data needed for distant supervision training/testing

    **BEFORE USING THIS**

        RENAME SAVE FILE!!!!


    ensure: 
    - features/stances of articles is up to date --->    ../resources//contentTrain//out
    - linguistic features are up to date --->     ../resources//dataFiles//lingFeatures//out/
"""


def OLDmakeDistantSupervisionData():
    # [[truthvalues], [{features}], [domain], [file_name]]
    fullDataSet = [[], [], [], []]
    
    for file_ in os.listdir("../resources//contentTrain//out"):
        if file_.endswith(".json"):
            with open("../resources//contentTrain//out/" + file_, 'r') as doc:
                claimData =  json.loads(doc.read())
            
            for article in claimData[1]:
                articleFeats = {}
                fullDataSet[0].append(claimData[0][0])#truth value
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

                articleFeats[2] = sumprobabilities[0]#prob support
                articleFeats[3] = sumprobabilities[1]#prob refute
                articleFeats.update(lFileDict[article[2]])#Use url as key in lingfeats
                intarticleFeats = {}
                for feat in articleFeats:
                    intarticleFeats[int(feat)] = articleFeats[feat]
                fullDataSet[1].append(intarticleFeats)#features
                fullDataSet[2].append(article[1]) #domain
                fullDataSet[3].append(file_)
    
    with open('../resources//properDistantSupervisionDataV3.pickle', 'wb') as handle:
        pickle.dump(fullDataSet, handle, protocol=pickle.HIGHEST_PROTOCOL)




"""
    CHANGE THE SAVE FILE BEFORE YOU RUN IT
"""
def makeDistantSupervisionData(infoAdr, lingfeatsAdr):
    # [[truthvalues], [{features}], [domain], [file_name]]
    fullDataSet = [[], [], [], []]

    for file_ in os.listdir(infoAdr):
        if file_.endswith(".json"):
            with open(infoAdr + "/" + file_, 'r') as doc:
                claimData =  json.loads(doc.read())
            
            for article in claimData[1]:
                articleFeats = {}
                fullDataSet[0].append(claimData[0][0])#truth value
                sumprobabilities = [0,0]

                #avg probabilities of first 5 snippets in article
                for i, snippet in enumerate(article[0][:5]):
                    sumprobabilities[0] += snippet[0][0]
                    sumprobabilities[1] += snippet[0][1]
                sumprobabilities[0] /= i+1
                sumprobabilities[1] /= i+1
                
                #get ling feats
                with open(lingfeatsAdr+"/"+file_ , 'r') as lFile:
                    lFileDict = json.load(lFile)

                articleFeats[2] = sumprobabilities[0]#prob support
                articleFeats[3] = sumprobabilities[1]#prob refute
                articleFeats.update(lFileDict[article[2]])#Use url as key in lingfeats
                intarticleFeats = {}
                for feat in articleFeats:
                    intarticleFeats[int(feat)] = articleFeats[feat]
                fullDataSet[1].append(intarticleFeats)#features
                fullDataSet[2].append(article[1]) #domain
                fullDataSet[3].append(file_)
    
    with open('../resources//wikiPeopleDistantSupervision.pickle', 'wb') as handle:
        pickle.dump(fullDataSet, handle, protocol=pickle.HIGHEST_PROTOCOL)



def testModel(dataSet, model):
    """
    return: list of two tuples containing the summary data of the testing
        [(wrongFalse, wrongTrue), (correctFalse, correctTrue)]

    param: dataSet: list containing [[truthvalues], [{features}]], truth values list can be empty
    param: model: trained/loaded liblinear model object to be used in the prediction

    """
    
    p_labels, p_acc, p_vals = llu.predict(dataSet[0], dataSet[1], model, '-b 1')

    # Load reliability
    # default dict, default -1
    with open("../resources//compiledReliabilityDict707.txt", "r") as relFile:
        relDict = json.load(relFile)
    relDict = defaultdict(lambda: -1, relDict)

    judgements = []
    current_claim = ""
    current_claim_truth = -1
    sumCurrentProbs = [0,0]
    for i, probs in enumerate(p_vals):#run through results adjusting for reliability and summarizing for full claims

        if (i>0) and (current_claim != dataSet[3][i]):#not first claim in dataset AND new claim
            if sumCurrentProbs[0] < sumCurrentProbs[1]:
                judgements.append((1, current_claim_truth))
            else:
                judgements.append((0,current_claim_truth))
            sumCurrentProbs = [0,0]
        
        #sum probabilities for the current claim
        current_claim_truth = dataSet[0][i]
        current_claim = dataSet[3][i]
        reliability = relDict[dataSet[2][i]]
        if reliability == -1:
            reliability = 0.5
        elif reliability < 0.1:
            reliability = 0.1
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

    print(f"claims = {i+1}")
    print(f"wrongFalse = {wrongFalse}")
    print(f"wrongTrue = {wrongTrue}")

    return [(wrongFalse, wrongTrue), (correctFalse, correctTrue)]


def training(dataSet, save = False):
    """
        trains the L1 logistic regresssion model

        return: trained liblinear model object

        param: dataSet: training data in form:
                    [[truthVales], [{features}]]
    """
    model = llu.train(dataSet[0], dataSet[1], '-s 6 -w1 5 -q') # -v 10

    if save:
        llu.save_model("../resources//models/distantSupervisionV2M2.model",model)

    return model


if __name__ == "__main__":
    # makeDistantSupervisionData('../resources//wikiData//outPeople', '../resources//wikiData//lingFout')
        
    with open('../resources//properDistantSupervisionDataV3.pickle', 'rb') as handle:
        dataSet = pickle.load(handle)

    dataSize = len(dataSet[0])

    wrong = [0,0]
    right = [0,0]

    for x in range(0,10):
        print(x+1)
        testStart = int(x*dataSize/10)
        testStop = int((x+1)*dataSize/10)
        print(testStart)
        print(testStop)
        one = [dataSet[0][0:testStart]+dataSet[0][testStop:], dataSet[1][0:testStart]+dataSet[1][testStop:], dataSet[2][0:testStart]+dataSet[2][testStop:], dataSet[3][0:testStart]+dataSet[3][testStop:]]
        two = [dataSet[0][testStart:testStop], dataSet[1][testStart:testStop], dataSet[2][testStart:testStop], dataSet[3][testStart:testStop]]
        model = training(one)
        tempw, tempr = testModel(two, model)
        wrong = np.add(wrong, tempw)
        right = np.add(right, tempr)
        print("**********************************")


    print(f"total correctFalse = {right[0]}")
    print(f"total correctTrue = {right[1]}")
    print(f"total wrongFalse = {wrong[0]}")
    print(f"total wrongTrue = {wrong[1]}")

    print(f"total accuracy = {(right[1]+right[0])/(wrong[0]+wrong[1]+right[1]+right[0])}")
    print(f"true accuracy {right[1]/(right[1]+wrong[1])}")
    print(f"false accuracy {right[0]/(right[0]+wrong[0])}")