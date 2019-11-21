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


"""
    Generates the final data needed for Joing model (CRF) training/testing
    Also trains and tests the model

    **BEFORE USING THIS**

    ensure: 
    - features/stances of articles is uptodaet --->    ../resources//contentTrain//out
    - linguistic features are up to date --->     ../resources//dataFiles//lingFeatures//out/
    - reliability file is upto date -----> ../resources//compiledReliabilityDict702.txt
"""

def makeJointData():
    # [[truthvalues], [{features}], [file_name]]
    fullDataSet = [[], [], []]
    
    #reliability data
    with open("../resources//compiledReliabilityDict702.txt", "r") as relFile:
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

                    articleFeats[2] = sumprobabilities[0]#prob support
                    articleFeats[3] = sumprobabilities[1]#prob refute
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

    
    with open('../resources//jointModelData.pickle', 'wb') as handle:
        pickle.dump(fullDataSet, handle, protocol=pickle.HIGHEST_PROTOCOL)


def trainJoint():
    with open('../resources//jointModelDataV1.pickle', 'rb') as handle:
        dataSet = pickle.load(handle)


    model = llu.train(dataSet[0][185:], dataSet[1][185:], '-s 6 -w1 2.7') # -v 10 

    llu.save_model("../resources//models/jointModelSample.model",model)


def testJoint():
    with open('../resources//jointModelDataV1.pickle', 'rb') as handle:
        dataSet = pickle.load(handle)
    
    model = llu.load_model("../resources//models/jointModelSample.model")
    p_labels, p_acc, p_vals = llu.predict(dataSet[0][:185], dataSet[1][:185], model, '-b 1')

    wrongTrue = 0
    wrongFalse = 0
    for i, x in enumerate(p_labels):
        if int(x) != int(dataSet[0][i]):
            if int(dataSet[0][i]) == 1:
                wrongTrue += 1
            else:
                wrongFalse += 1
    print(f"wrongFalse = {wrongFalse}")
    print(f"wrongTrue = {wrongTrue}")         



if __name__ == "__main__":
    # makeJointData()
    # trainJoint()
    testJoint()