import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

import json
from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from util import featureBag
import pickle

#dictionary containing all possible features 
featureDict = featureBag.getFeatureFile("../resources//featsFull.pickle")
# reliabilityDict = featureBag.getFeatureFile("../resources/reliability.pickle")

dataSet = [[],[]]

# #load data from all source files
# for file_ in os.listdir("../resources//contentTrain//output"):
#     if file_.endswith(".json"):
#         with open("../resources//contentTrain//output/" + file_, 'r') as doc:#read snopes file
#             fileData =  json.loads(doc.read())
#         listofProperDicts = []
#         for dict_ in fileData[1]:
#             newDict = {}
#             for key in dict_:
#                 newDict[int(key)] = dict_[key]
#             listofProperDicts.append(newDict)

#         dataSet[0].extend(fileData[0])
#         dataSet[1].extend(listofProperDicts)    

    
# with open('contentAwareData.pickle', 'wb') as handle:
#     pickle.dump(dataSet, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('distantSupervisionData.pickle', 'rb') as handle:
    dataSet = pickle.load(handle)

model = llu.train(dataSet[0], dataSet[1], '-s 6 -w1 2.7 -v 10')
# llu.save_model("../resources//models/contentAware.model",model)

