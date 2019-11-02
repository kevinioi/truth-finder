import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
import json
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from liblinearpkg import *
from liblinearpkg import liblinearutil as llu

# dataSet = ([1,1,1,1,0,0,0,0], [])
# truthValue = None

# dataSet[1].append({1:20, 2:2, 3:23, 4:0.5})
# dataSet[1].append({1:18, 2:0, 3:24, 4:1.5})
# dataSet[1].append({1:30, 2:4, 3:42, 4:0.25})
# dataSet[1].append({1:24, 2:0.5, 3:32, 4:3})

# dataSet[1].append({1:4,2:65,3:2,4:50})
# dataSet[1].append({1:7,2:99,3:5,4:45})
# dataSet[1].append({1:2,2:67,3:1,4:24})
# dataSet[1].append({1:5,2:89,3:3,4:37})

# model = llu.train(dataSet[0], dataSet[1], '-c 11')

# llu.predict([1],[{1:45,2:2,3:54,4:1.5}],model)


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
        features = []
        for gram in unigrams:
            features.append(gram)
        for gram in bigrams:
            features.append(gram)
        dataSet[1].append(features)


# model = llu.train(dataSet[0], dataSet[1], '-c 11')

# llu.predict([1],[{1:45,2:2,3:54,4:1.5}],model)

# llu.save_model("testModel", model)