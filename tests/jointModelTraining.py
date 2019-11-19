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


with open('compiledReliabilityDict502.txt', 'rb') as handle:
    reliabilityDict = json.load(handle)

dataSet = [[],[]]

"""
*************************************************
            Haven't implemented Reliability Yet
*************************************************
"""

# load data from all source files
# for file_ in os.listdir("../resources//contentTrain//output"):
#     if file_.endswith(".json"):
#         claimDict = defaultdict(lambda: 0)
#         with open("../resources//contentTrain//output/" + file_, 'r') as doc:#read snopes file
#             fileData =  json.loads(doc.read())
#         for i, dict_ in enumerate(fileData[1]):
#             for key in dict_:
#                 claimDict[int(key)] += dict_[key]
#             try:
#                 claimDict[606568] = int(100*reliabilityDict[fileData[2][i]])#***************************Insert reliability stuff here
#             except:
#                 claimDict[606568] = 50
#                 pass
#         dataSet[0].append((fileData[0])[0])
#         dataSet[1].append(dict(claimDict))

    
# with open('jointModelTrainingData.pickle', 'wb') as handle:
#     pickle.dump(dataSet, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('jointModelTrainingData.pickle', 'rb') as handle:
    dataSet = pickle.load(handle)



model = llu.train(dataSet[0], dataSet[1], '-s 6 -w1 2.7 -v 10')
# llu.save_model("../resources//models/contentAware.model",model)

