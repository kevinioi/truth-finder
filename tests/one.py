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


myDict = featureBag.getFeatureFile("../resources/feats.pickle")


# print(myDict[('spoof',)])

for i in myDict.keys():
    if myDict[i] == 317:
        print(i)


# for i, x in enumerate(myDict):
#     print(f'{x}: {myDict[x]}')
#     if i > 30:
#         break
# print(myDict[("review",)])

# featureBag.createFeatureFile('../../data/', '../resources/feats.pickle')

"""
    count refute and support
"""

# true = 0
# false = 0

# for file_ in os.listdir("../..//data"):
#     if file_.endswith(".json"):
#         with open("../..//data/" + file_, 'r') as doc:
#             fileData =  json.loads(doc.read())

#         if fileData['Credibility'] == 'false' or fileData['Credibility'] == 'mostly false':
#             false +=1
#         elif fileData['Credibility'] == 'true' or fileData['Credibility'] == 'mostly true':
#             true += 1

# print(true)
# print(false)
# print(false/true)