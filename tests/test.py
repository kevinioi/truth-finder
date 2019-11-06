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
import pickle

#dictionary containing all possible features 
# featDict = featureBag.getFeatureFile("../resources/featsV2.pickle")

def zez():
    return [[],[]]

test = defaultdict(lambda : [0,0])


print(test["Kevin"])


#load data from all source files
# for file_ in os.listdir("../resources//snopesData"):
    
#     truthValue = None

#     if file_.endswith(".json"):
#         with open("../resources//snopesData/" + file_, 'r') as doc:
#             fileData =  json.loads(doc.read())

#         if fileData['Credibility'] == 'false' or fileData['Credibility'] == 'mostly false':
#             truthValue = 0
#         elif fileData['Credibility'] == 'true' or fileData['Credibility'] == 'mostly true':
#             truthValue = 1

#         for page in fileData["Google Results"]:
#             print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#             for sourceDict in page.values():
#                 for source in sourceDict:
#                     print("**************************")
#                     print(source["domain"])
#                     print(source["link"])
#     break

