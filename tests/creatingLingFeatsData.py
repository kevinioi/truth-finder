import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

import json
from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from util import featureBag
import pickle
from collections import defaultdict
from multiprocessing import Pool

def getLingFeats(dirAdr):
"""
 give it ../resources//dataFiles//lingFeatures//one, two, three...

 it will grab all relevent articles from the contentTrain//output dir
    - get the url and pull the lingfeats from the article 
"""


    with open('correctedLingfeats.pickle', 'rb') as handle:
        lingFeatureDict =  pickle.load(handle)
    lingFeatureDict = defaultdict(lambda: 0, linguisticFeatures)

    for file_ in os.listdir(dirAdr):
        if file_.endswith(".json"):
            claimLingFeats = {}

            with open("../resources//contentTrain//output/" + file_, 'r') as doc:
                claimData =  json.loads(doc.read())
            for article in claimData[1]:
                url = article[1][1]
                articleText = textProcessor.pullArticleText(url ,timeoutTime=6)
                releventFeatures = prepArticleForClassification(articleText, lingFeatDict)
                claimLingFeats[url] = releventFeatures

            with open("../resources//dataFiles//lingFeatures//out/"+ file_) as outputFile:
                json.dump( claimLingFeats, outputFile)


def makeDistantSupervisionData():

    # [[truthvalues], [{features}], [domain], [file_name]]
    fullDataSet = [[], [], [], []]
    
    for file_ in os.listdir("../resources//contentTrain//output"):
        if file_.endswith(".json"):
            with open("../resources//contentTrain//output/" + file_, 'r') as doc:
                claimData =  json.loads(doc.read())
            
            for article in claimData[1]:
                articleFeats = {}
                fullDataSet[0].append(claimData[0][0])#truth value
                sumprobabilities = [0,0]

                #avg probabilities of first 5 snippets in article
                for i, snippet in enumerate(article[0][:5]):
                    sumprobabilities[0] += snippet[0[0]]
                    sumprobabilities[1] += snippet[0[1]]
                sumprobabilities[0] /= i+1
                sumprobabilities[1] /= i+1
                
                #get ling feats
                with open("../resources//dataFiles//lingFeatures//out/"+file_ 'r') as lFile:
                    lFileDict = json.load(lFile)

                articleFeats[2] = sumprobabilities[0]#prob support
                articleFeats[3] = sumprobabilities[1]#prob refute
                articleFeats.update(lFileDict[article[2]])#Use url as key in lingfeats
                fullDataSet[1].append(articleFeats)#features
                fullDataSet[2].append(article[1]) #domain
                fullDataSet[3].append(file_)
    
    with open('properDistantSupervisionData.pickle', 'wb') as handle:
        pickle.dump(fullDataSet, handle, protocol=pickle.HIGHEST_PROTOCOL)


def makeJointData():

    # [[truthvalues], [{features}], [domain], [file_name]]
    fullDataSet = [[], [], []]
    
    # Load reliability
    with open("compiledReliabilityDict502.txt", "r") as relFile:
        relDict = json.load(relFile)
    relDict = defaultdict(lambda: -1, relDict)

    for file_ in os.listdir("../resources//contentTrain//output"):
        if file_.endswith(".json"):
            aggregateClaimInfo = [[],[],[]]
            with open("../resources//contentTrain//output/" + file_, 'r') as doc:
                claimData =  json.loads(doc.read())
            
            """
            compile individual article information
            """
            for article in claimData[1]:
                articleFeats = {}
                aggregateClaimInfo[0].append(claimData[0][0])#truth value
                sumprobabilities = [0,0]

                #avg probabilities of first 5 snippets in article
                for i, snippet in enumerate(article[0][:5]):
                    sumprobabilities[0] += snippet[0[0]]
                    sumprobabilities[1] += snippet[0[1]]
                sumprobabilities[0] /= i+1
                sumprobabilities[1] /= i+1
                
                #get linguistic features for article
                with open("../resources//dataFiles//lingFeatures//out/"+file_ 'r') as lFile:
                    lFileDict = json.load(lFile)

                articleFeats[2] = sumprobabilities[0]#prob support
                articleFeats[3] = sumprobabilities[1]#prob refute
                articleFeats.update(lFileDict[article[2]])#Use url as key in lingfeats
                aggregateClaimInfo[1].append(articleFeats)#features
                aggregateClaimInfo[2].append(relDict[article[1]]) #domain
            

            """
            compile generalized claim features
            """
            claimFeatures = defaultdict(lambda: 0)
            reliability = 0.0
            truthValue = aggregateClaimInfo[0][0]
            for article in aggregateClaimInfo:
                for i, featureDict in enumerate(aggregateClaimInfo[1]):
                    for feature in featureDict:
                        claimFeatures[feature] += featureDict[feature]
                    if aggregateClaimInfo[2][i] == -1:
                        reliability += 0.5
                    elif aggregateClaimInfo[2][i] < 0.25:
                        reliability += 0.25
                    else:
                        reliability += aggregateClaimInfo[2][i]

            fullDataSet[0].append(truthValue)
            fullDataSet[1].append(dict(claimFeatures))
            fullDataSet[2].append(file_)
    
    with open('properJointData.pickle', 'wb') as handle:
        pickle.dump(fullDataSet, handle, protocol=pickle.HIGHEST_PROTOCOL)

                

if __name__ == "__main__":
    
    with Pool(processes=6) as pool:
        procs = []
        procs.append(pool.apply_async(getLingFeats,("../resources//dataFiles//lingFeatures//one",)))
        procs.append(pool.apply_async(getLingFeats,("../resources//dataFiles//lingFeatures//two",)))
        procs.append(pool.apply_async(getLingFeats,("../resources//dataFiles//lingFeatures//three",)))
        procs.append(pool.apply_async(getLingFeats,("../resources//dataFiles//lingFeatures//four",)))
        procs.append(pool.apply_async(getLingFeats,("../resources//dataFiles//lingFeatures//five",)))
        procs.append(pool.apply_async(getLingFeats,("../resources//dataFiles//lingFeatures//six",)))

        #wait for each process to finish
        for proc in procs:
            proc.wait()

