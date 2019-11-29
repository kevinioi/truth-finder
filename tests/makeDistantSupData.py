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
    Generates the final data needed for distant supervision training/testing

    **BEFORE USING THIS**

        RENAME SAVE FILE!!!!


    ensure: 
    - features/stances of articles is up to date --->    ../resources//contentTrain//out
    - linguistic features are up to date --->     ../resources//dataFiles//lingFeatures//out/
"""
maxArticleCount = 30

def main():
    # makeDistantSupervisionData('../resources//wikiData//outPeople', '../resources//wikiData//lingFout')
    # makeDistantSupervisionData('../resources//contentTrain//out', '../resources//dataFiles//lingFeatures//out')
        
    # with open('../resources//pureSocialData.pickle', 'rb') as handle:
    with open('../resources//properDistantSupervisionDataV3.pickle', 'rb') as handle:
        dataSet = pickle.load(handle)

    normFDict = defaultdict(lambda: 0)
    for fdict in dataSet[1]:
        for feature in fdict:
            if normFDict[feature] < fdict[feature]:
                normFDict[feature] = fdict[feature]
    
    for i in range(len(dataSet[1])):
        for feature in dataSet[1][i]:
            dataSet[1][i][feature] /= normFDict[feature] 

    """
        REMOVES LINGUISTIC FEATURES
    """
    # for i in range(len(dataSet[1])):
    #     cleanDict = {}
    #     cleanDict[2] = dataSet[1][i][2]
    #     cleanDict[3] = dataSet[1][i][3]
    #     dataSet[1][i] = cleanDict

    """
        REMOVES STANCE PROBABILITIES
    """
    # for i in range(len(dataSet[1])):
    #     dataSet[1][i][2] = 0
    #     dataSet[1][i][3] = 0

    wrong = [0,0]
    right = [0,0]
    auc = 0
    #10-fold cross-validation
    for x in range(0,10):
        print(x+1)
        testStart = int(x*dataSize/10)
        testStop = int((x+1)*dataSize/10)
        print(testStart)
        print(testStop)
        one = [dataSet[0][0:testStart]+dataSet[0][testStop:], dataSet[1][0:testStart]+dataSet[1][testStop:], dataSet[2][0:testStart]+dataSet[2][testStop:], dataSet[3][0:testStart]+dataSet[3][testStop:]]
        two = [dataSet[0][testStart:testStop], dataSet[1][testStart:testStop], dataSet[2][testStart:testStop], dataSet[3][testStart:testStop]]
        model = training(one)
        tempw, tempr, tempauc = testModel(two, model)
        wrong = np.add(wrong, tempw)
        right = np.add(right, tempr)
        auc += tempauc
        print("**********************************")

    print(f"Avg AUC = {auc/10}")
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

def makeDistantSupervisionData(infoAdr, lingfeatsAdr):
    """
        CHANGE THE SAVE FILE BEFORE YOU RUN IT
        
        make sure these files are up to date:
        - features/stances of articles is up to date --->    ../resources//contentTrain//out
        - linguistic features are up to date --->     ../resources//dataFiles//lingFeatures//out/
    """
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
    
    with open('../resources//properDistantSupervisionDataV4.pickle', 'wb') as handle:
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
    with open("../resources//compiledReliabilityDictFINAL.txt", "r") as relFile:
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
        # sumCurrentProbs[0] += probs[0]
        # sumCurrentProbs[1] += probs[1]

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


def training(dataSet, save = False):
    """
        trains the L1 logistic regresssion model

        return: trained liblinear model object

        param: dataSet: training data in form:
                    [[truthVales], [{features}]]
    """
    # model = llu.train(dataSet[0], dataSet[1], '-s 6 -w1 3.7 -q') # FOR FULL DISTANT SUPERVISION
    # model = llu.train(dataSet[0], dataSet[1], '-s 6 -w1 3 -q') # FOR (STANCE + LINGUISTICS) & (RELIABILITY + STANCE)


    ratio = 1/(sum(dataSet[0])/len(dataSet[0]))
    model = llu.train(dataSet[0], dataSet[1], f'-s 6 -w1 {ratio} -q') # FOR RELIABILITY + STANCE



    if save:
        llu.save_model(f"../resources//models/{save}.model",model)

    return model

def getLongTail(maxArticleCount):

      # [[truthvalues], [{features}], [domain], [file_name]]
    longTailData = [[],[],[],[]]

    for file_ in os.listdir("../resources//contentTrain//out"):
        articleCount = 0
        if file_.endswith(".json"):
            with open("../resources//contentTrain//out/" + file_, 'r') as doc:
                claimData =  json.loads(doc.read())

            articleCount = len(claimData[1])
            
            if articleCount > 0 and articleCount <= maxArticleCount:
                for article in claimData[1]:
                    articleFeats = {}
                    longTailData[0].append(claimData[0][0])
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
                    intarticleFeats = {}
                    for feat in articleFeats:
                        intarticleFeats[int(feat)] = articleFeats[feat]
                    longTailData[1].append(intarticleFeats)#features
                    longTailData[2].append(article[1]) #domain
                    longTailData[3].append(file_)
    
    return longTailData

if __name__ == "__main__":
    main()
