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

#dictionary containing all possible features 
featureDict = featureBag.getFeatureFile("../resources//featsFull.pickle")
# reliabilityDict = featureBag.getFeatureFile("../resources/reliability.pickle")

dataSet = [[],[]]


"""
*************************************************
            Haven't implemented Reliability Yet
*************************************************
"""

# load data from all source files
for file_ in os.listdir("../resources//contentTrain//output"):
    if file_.endswith(".json"):
        claimDict = defaultdict(lambda: 0)
        with open("../resources//contentTrain//output/" + file_, 'r') as doc:#read snopes file
            fileData =  json.loads(doc.read())
        for dict_ in fileData[1]:
            for key in dict_:
                claimDict[int(key)] += dict_[key]
        claimDict[0] = 0.5#***************************Insert reliability stuff here

        dataSet[0].append((fileData[0])[0])
        dataSet[1].append(dict(claimDict))

    
with open('jointData.pickle', 'wb') as handle:
    pickle.dump(dataSet, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('jointData.pickle', 'rb') as handle:
#     dataSet = pickle.load(handle)


# model = llu.train(dataSet[0], dataSet[1], '-s 6 -w1 2.7 -v 10')
# llu.save_model("../resources//models/contentAware.model",model)

