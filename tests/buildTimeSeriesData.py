import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

import json
from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from util import featureBag
from util import textProcessor
from collections import defaultdict
from multiprocessing import Pool
import pickle


"""
    takes raw snopes files from ../resources//contentTrain//one, two, three
    - Pulls linguistic features, puts them under filename in ../resources//dataFiles//lingFeatures//out/
    - Pulls stance class, truth values, domain, url, stores it under filename in ../resources//contentTrain//out/
"""

def buildData(dirAdr):
    features = featureBag.getFeatureFile("../resources/stanceFeatsV2.pickle")
    model = llu.load_model("../resources/models/stance2v2.model")

    with open('correctedLingfeats.pickle', 'rb') as handle:
        linguisticFeatures =  pickle.load(handle)
    linguisticFeatures = defaultdict(lambda: 0, linguisticFeatures)
    
    with open("../resources//compiledReliabilityDict707.txt", "r") as relFile:
        relDict = json.load(relFile)
    relDict = defaultdict(lambda: -1, relDict)

    with open("../resources//timeSeries//timeSeriesLinks.txt", "r") as sourceFile:
        timeSeriesLinks = json.load(sourceFile)
        
    #will hold all the info
    fullData = {}

    #load data from all files in directory
    for claim in timeSeriesLinks:
                        #dictionary of days and articles, truthValue
        fullData[claim] = [{},int(timeSeriesLinks[claim][1])]

        #I FORGOT TO STORE THE TEXTUAL CLAIM
        #FIND IT IN THE SOURCE FILES USING CLAIM FILENAME

        textualClaim = ""
        for sourceClaim in os.listdir("../resources//contentTrain//used_claims"):
            if sourceClaim == claim:
                with open("../resources//contentTrain/used_claims/" + sourceClaim, 'r') as doc:#read snopes file
                    sourceClaimText = json.loads(doc.read())
                textualClaim = sourceClaimText["Claim"]
                        
        for day in timeSeriesLinks[claim][0]:
            fullData[claim][0][day] = []

            for article in timeSeriesLinks[claim][0][day]:
                # holds all of the linguistic features present in the article and the domain
                articleInfo = [{}, ""]
                
                try:
                    text = textProcessor.pullArticleText(article[0],timeoutTime=20)
                    snippets = textProcessor.getSnippets(text, 4)
                    releventSnips = textProcessor.getRelevence(textualClaim,snippets)
                    numRelevent = len(releventSnips[0])                  

                    if numRelevent > 0:

                        """
                            GET LINGUISTIC FEATURES
                        """
                        lingFeats = textProcessor.prepArticleForClassification(text, linguisticFeatures)

                        #add linguistic features to article feature dictionary
                        for feature in lingFeats:
                            articleInfo[0][int(feature)] = lingFeats[feature] 
                        """
                        """

                        snipData = textProcessor.prepListForClassification(releventSnips[0],features)
                        p_labels, p_acc, p_vals = llu.predict( [], snipData, model, '-b 1 -q')

                        #get stance*overlap
                        stanceImpact = []
                        for index, probVals in enumerate(p_vals):
                            probs = [[0,0], releventSnips[0][index]] # [[overlap*probS, overlap*prob], snippet]
                            probs[0][0] = (releventSnips[1])[index]*probVals[0]
                            probs[0][1] = (releventSnips[1])[index]*probVals[1]
                            stanceImpact.append(probs)
                        #sort by max overlap*probStance
                        stanceImpact.sort(key= lambda instance: max(instance[0][0], instance[0][1]),reverse=True)

                        #average top 5 stance positions
                        avgStanceProbs = [0,0]
                        for i, probs in enumerate(stanceImpact[:5]):
                            avgStanceProbs[0] += probs[0][0]
                            avgStanceProbs[1] += probs[0][1]
                        avgStanceProbs[0] /= i+1
                        avgStanceProbs[1] /= i+1
                        articleInfo[0][2] = avgStanceProbs[0]
                        articleInfo[0][3] = avgStanceProbs[1]

                        #get reliability
                        reliability = relDict[article[1]]
                        if reliability == -1:#new domian
                            reliability = 0.5
                        elif reliability < 0.15:#baseline reliability
                            reliability = 0.15
                        articleInfo[0][1] = reliability
                        articleInfo[1] = (article[1], reliability)

                        #add article information to the days info
                        fullData[claim][0][day].append(articleInfo)
                except: # Exception as e
                    # raise e
                    continue
            
    with open("../resources//timeSeries//out/fullData.json", "w") as fp:
        json.dump(fullData, fp)



def parseFile(fileAdr):
    info = {}
    currentFile = None
    currentDay = 0
    with open(fileAdr, "r") as sourceFile:
        for line in sourceFile.readlines():
            text = line.split()

            if text[0].strip() == ">>": #new claim
                currentFile = text[1]
                truthValue = text[2]
                info[currentFile] = [{}, truthValue]
            elif text[0].strip() == "*": #new day
                currentDay = text[1]
                info[currentFile][0][currentDay] = []
            elif text[0].strip() == "-": #new source
                info[currentFile][0][currentDay].append([text[1], text[2]])

    with open("timeSeriesLinks.txt", "w") as file_:
        json.dump(info, file_)

# claimDict =    
# {
# claim1 =
#     [
#         {
#             day1 = 
#             {
#                 [
#                     [[features], reliability, domain]
#                 ]
#             }
#         }
#         ,
#         truthValue
#     ]
# }


if __name__ == "__main__":
    
    buildData("timeSeriesLinks.txt")
    # parseFile("TimeSeries.txt")

    # with Pool(processes=6) as pool:
    #     procs = []
    #     procs.append(pool.apply_async(buildData,("../resources//contentTrain//one",)))
    #     procs.append(pool.apply_async(buildData,("../resources//contentTrain//two",)))
    #     procs.append(pool.apply_async(buildData,("../resources//contentTrain//three",)))
    #     procs.append(pool.apply_async(buildData,("../resources//contentTrain//four",)))
    #     procs.append(pool.apply_async(buildData,("../resources//contentTrain//five",)))
    #     procs.append(pool.apply_async(buildData,("../resources//contentTrain//six",)))
    #     #wait for each process to finish
    #     for proc in procs:
    #         proc.wait()