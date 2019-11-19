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

#dictionary containing all possible features 
featureDict = featureBag.getFeatureFile("../resources//featsFull.pickle")
# reliabilityDict = featureBag.getFeatureFile("../resources/reliability.pickle")

count = 0
#load data from all source files
# for file_ in os.listdir("../resources//contentTrain"):
for file_ in os.listdir("../resources//contentTrain"):
    count+=1
    print("************************************")
    print(file_)
    print(count)
    print("************************************")

    if file_.endswith(".json"):
        with open("../resources//contentTrain/" + file_, 'r') as doc:#read snopes file
        # with open("../resources//wikiData//WikiHoaxes/" + file_, 'r', encoding='utf-8') as doc:#read snopes file
            fileData =  json.loads(doc.read())
        
        if fileData['Credibility'] == 'false' or fileData['Credibility'] == 'mostly false':
            truthValue = 0
        elif fileData['Credibility'] == 'true' or fileData['Credibility'] == 'mostly true':
            truthValue = 1
        
        #infolist = [[truthValue, ],[articleFeatures, ], [reliability]]
        infolist = [[], [], []]

        """
        ****************
        Normal claim file processing
        ****************
        """

        for page in fileData["Google Results"]:#load page of google results
            for resultsDict in page.values():#load sources from google page
                for source in resultsDict:#process each source
                    if (source["domain"] != "www.snopes.com"):
                        print(source["domain"])
                        try:
                            wordBagList =  textProcessor.pullArticleText(source["link"],timeoutTime=6)
                            infolist[2].append(source["domain"])
                            infolist[1].append(textProcessor.prepArticleForClassification(wordBagList, featureDict))  
                            infolist[0].append(truthValue)
                        except:
                            continue

            """
            going through wikiHoaxes
            """
        # for page in fileData["Google Results"]:#load page of google results
        #     # for resultsDict in page.values():#load sources from google page
        #     for resultsDict in page["results"]:#load sources from google page
        #         print(resultsDict["domain"])
        #         try:
        #             wordBagList =  textProcessor.pullArticleText(resultsDict["link"],timeoutTime=6)
        #             infolist[2].append(resultsDict["domain"])
        #             infolist[1].append(textProcessor.prepArticleForClassification(wordBagList, featureDict))  
        #             infolist[0].append(truthValue)
        #         except:
        #             continue
                            
        with open("../resources//contentTrain//output/"+file_, "w") as claimFile:
            json.dump(infolist, claimFile)
                            