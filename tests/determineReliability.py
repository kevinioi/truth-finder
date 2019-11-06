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
from util import textProcessor
from collections import defaultdict
import pickle

#dict to store results of correct/incorrect stance classifications during training
# key will be web domains
# initialize values to [0 correct, 0 incorrect] on new domain
reliability = defaultdict(lambda : [0,0])

featDict = featureBag.getFeatureFile("../resources/featsV2.pickle")
model = llu.load_model("../resources/models/gen2v1.model")

#load data from all source files
for file_ in os.listdir("../resources//snopesData"):    
    truthValue = None

    if file_.endswith(".json"):
        with open("../resources//snopesData/" + file_, 'r') as doc:
            fileData =  json.loads(doc.read())

        if fileData['Credibility'] == 'false' or fileData['Credibility'] == 'mostly false':
            truthValue = 0
        elif fileData['Credibility'] == 'true' or fileData['Credibility'] == 'mostly true':
            truthValue = 1

        for page in fileData["Google Results"]:
            for resultsDict in page.values():
                for source in resultsDict:
                        try:
                            text = textProcessor.pullArticleText(source["link"])
                            snippets = textProcessor.getSnippets(text, 4, fileData["Claim"])
                            probSum = [0,0]

                            print(snippets)
                            for s in snippets:
                                p_labels, p_acc, p_vals = llu.predict( [], [textProcessor.prepTextForClassification(s,featDict)], model, '-b 1')
                                probSum[0] += (p_vals[0])[0]
                                probSum[1] += (p_vals[0])[1]

                            #check if the stance of the article aligns with known truth value of claim
                            if (probSum[truthValue] > probSum[1]):
                                (reliability[source["domain"]])[0] += 1#correct
                            else:
                                (reliability[source["domain"]])[1] += 1#incorrect
                        except:
                            continue
                break
            break
        break
    break
print(reliability)

"""
****************************************************************************************************************
"""

# for claim in file:
#     #default dict
#     reliability[domain] = {[correct, incorrect]}
#                 'www.wiki.com' , [5, 8]
#     sources[] = pullwebSources

#     truthValue = claim.truth

#     for source in sources:
#         data = pullData
#         snips = data.getReleventSnips
#         stance = avgStance(snips)

#         if stance == truthValue:
#             (reliability[getCleanUrl(source)])[0] += 1
#         else:
#             (reliability[getCleanUrl(source)])[1] += 1


# calcedReliability = {}

# for source in reliability.keys():
#     calcedReliability[domain] = (reliability[source])[0]/((reliability[source])[0] +(reliability[source])[1])
