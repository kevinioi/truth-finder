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


"""
combine dicts
"""
# count = 0
# fullDict = defaultdict(lambda : 0)


# with open('lingFeats1.pickle', 'rb') as handle:
#     dataSet = pickle.load(handle)

# for word in dataSet:
#     if fullDict[word] == 0:
#         count +=1
#         fullDict[word] = count


# with open('lingFeats2.pickle', 'rb') as handle:
#     dataSet = pickle.load(handle)
# for word in dataSet:
#     if fullDict[word] == 0:
#         count +=1
#         fullDict[word] = count

# with open('lingFeats3.pickle', 'rb') as handle:
#     dataSet = pickle.load(handle)
# for word in dataSet:
#     if fullDict[word] == 0:
#         count +=1
#         fullDict[word] = count

# with open('lingFeats4.pickle', 'rb') as handle:
#     dataSet = pickle.load(handle)
# for word in dataSet:
#     if fullDict[word] == 0:
#         count +=1
#         fullDict[word] = count

# print(len(dataSet))


# with open('lingFeatsComplete.pickle', 'wb') as handle:
#     pickle.dump(dict(fullDict), handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
Compose dict of words

feat1 -> words.txt
feat2 -> negative
feat3 -> positive
feat4 -> Affective
"""
# with open("../resources//linguisticLexicon/SampleAffectiveFeats.txt", "r") as file_:
#     content = file_.readlines()

# dd = defaultdict(lambda : 0)

# count=0

# for i,line in enumerate(content):
#     words = line.split(",")
#     for word in words:
#         if dd[word.strip()] == 0:
#             count+=1
#             dd[word.strip()] = count

# # print(dd)

# with open('lingFeats4.pickle', 'wb') as handle:
#     pickle.dump(dict(dd), handle, protocol=pickle.HIGHEST_PROTOCOL)
   

"""
check dicts
"""

with open('lingFeatsComplete.pickle', 'rb') as handle:
    dataSet = pickle.load(handle)

print(len(dataSet))

for x in dataSet:
    print(x)
    print(dataSet[x])
    break


"""
    count refute and support
"""

# true = 0
# false = 0

# for file_ in os.listdir("../resources//snopesData"):
#     if file_.endswith(".json"):
#         with open("../resources//snopesData/" + file_, 'r') as doc:
#             fileData =  json.loads(doc.read())

#         if fileData['Credibility'] == 'false' or fileData['Credibility'] == 'mostly false':
#             false +=1
#         elif fileData['Credibility'] == 'true' or fileData['Credibility'] == 'mostly true':
#             true += 1

# print(true)
# print(false)
# print(false/true)