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


#load pre-trained models
model1 = llu.load_model("../resources//models/gen2v1.model")
# model2 = llu.load_model("../resources//stance_models/modelWeight.liblin")

featDict = featureBag.getFeatureFile("../resources/featsV2.pickle")


# dataSet = ([], [])
# truthValue = None


# #load data from all source files
# for file_ in os.listdir("../resources//snopesData"):
#     if file_.endswith(".json"):
#         with open("../resources//snopesData/" + file_, 'r') as doc:
#             fileData =  json.loads(doc.read())

#         if fileData['Credibility'] == 'false' or fileData['Credibility'] == 'mostly false':
#             truthValue = 0
#         elif fileData['Credibility'] == 'true' or fileData['Credibility'] == 'mostly true':
#             truthValue = 1

#         wordBag = word_tokenize(fileData['Description'])
#         wordBag = [word.lower() for word in wordBag if word.isalpha()]
#         wordBagTagged = nltk.pos_tag(wordBag)

#         #remove named entities (eg. 'Trump','Obama') to prevent bias
#         wordBag = [word[0] for word in wordBagTagged if ((word[1] != 'NNP') & (word[1] != 'NNPS'))]

#         #get uni/bigrams
#         unigrams = nltk.ngrams(wordBag,n=1)
#         bigrams = nltk.ngrams(wordBag,n=2)
        
#         #add data to training/testing Sample
#         dataSet[0].append(truthValue)
#         features = defaultdict(featureBag.defDictFunc())
#         for gram in unigrams:
#             if featDict[gram] != 0:#don't count gram that aren't known features
#                 features[featDict[gram]] += 1
#         for gram in bigrams:
#             if featDict[gram] != 0:#don't count gram that aren't known features
#                 features[featDict[gram]] += 1
#         dataSet[1].append(dict(features))
#     break

with open('../resources/dataSetV2.pickle', 'rb') as handle:
    dataSet = pickle.load(handle)

print("****** Unweighted *******")
p_labels, p_acc, p_vals = llu.predict(dataSet[0], dataSet[1],model1, '-b 1 -p')

print(f"p_labels: {p_labels}  p_acc: {p_acc}   p_vals:  {p_vals}")

# print("****** Weighted *******")
# p_labels, p_acc, p_vals = llu.predict(dataSet[0], dataSet[1],model2, '-b 1')
# print(f"p_labels: {p_labels}  p_acc: {p_acc}   p_vals:  {p_vals}")


