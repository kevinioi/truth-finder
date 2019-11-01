import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
import json
from nltk.tokenize import word_tokenize
import nltk
import numpy as np


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
        for gram in unigrams:
            dataSet[1].append(gram)
        for gram in bigrams:
            dataSet[1].append(gram)

# for x in (dataSet[1])[:20]:
#     print(x)

        # for i, x in enumerate(bigrams):
        #     print(x)
        #     if i == 15:
        #         break

