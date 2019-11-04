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

#dictionary containing all possible features 
featDict = featureBag.getFeatureFile("../resources/feats.pickle")

dataSet = ([], [])
truthValue = None

#load data from all source files
for file_ in os.listdir("../..//data"):
    if file_.endswith(".json"):
        with open("../..//data/" + file_, 'r') as doc:
            fileData =  json.loads(doc.read())

        if fileData['Credibility'] == 'false':
            truthValue = 0
        else:
            truthValue = 1

        wordBag = word_tokenize(fileData['Description'])
        wordBag = [word.lower() for word in wordBag if word.isalpha()]
        wordBagTagged = nltk.pos_tag(wordBag)

        #remove named entities (eg. 'Trump','Obama') to prevent bias
        wordBag = [word[0] for word in wordBagTagged if ((word[1] != 'NNP') & (word[1] != 'NNPS'))]

        #get uni/bigrams
        unigrams = nltk.ngrams(wordBag,n=1)
        bigrams = nltk.ngrams(wordBag,n=2)
        
        #add data to training/testing Sample
        dataSet[0].append(truthValue)
        features = {}
        
        for value in featDict.values():
            features[value] = 0

        for gram in unigrams:
            if featDict[gram] != 0:#don't count gram that aren't known features
                features[featDict[gram]] += 1
        for gram in bigrams:
            if featDict[gram] != 0:#don't count gram that aren't known features
                features[featDict[gram]] += 1
        dataSet[1].append(dict(features))
    

model = llu.train(dataSet[0], dataSet[1], '-s 0 -w1 4')
# model = llu.train(dataSet[0], dataSet[1], '-s 0 -wi 4')
# model = llu.train(dataSet[0], dataSet[1], '-s 0 -v 10')

llu.save_model("modelWeight.liblin",model)

#llu.predict([1],[{1:45,2:2,3:54,4:1.5}],model)

# llu.save_model("testModel", model)