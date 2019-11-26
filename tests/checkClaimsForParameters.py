import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

import json
from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from util import featureBag
import pickle
from collections import defaultdict
from multiprocessing import Pool
from util import textProcessor


"""

    Browsing through contentTrain out to check for the subset test categories

"""

def getSocialsDataSet():
    """
        contructs dataSet containing only claims with more than 2 social media 
    """
        # [[truthvalues], [{features}], [domain], [file_name]]
    socialData = [[],[],[],[]]

    socials = ("facebook", "twitter", "quora", "reddit", "wordpress", "blogspot", "tumblr", "pinterest", "wikia")

    for file_ in os.listdir("../resources//contentTrain//out"):
        articleCount = 0
        if file_.endswith(".json"):
            with open("../resources//contentTrain//out/" + file_, 'r') as doc:
                claimData =  json.loads(doc.read())


            for article in claimData[1]:
                for x in socials:
                    if x in article[1]:
                        articleCount += 1
            
            if articleCount > 2:
                for article in claimData[1]:
                    for x in socials:
                        if x in article[1]:
                            articleFeats = {}
                            socialData[0].append(claimData[0][0])
                            sumprobabilities = [0,0]

                            #avg probabilities of first 5 snippets in article
                            for i, snippet in enumerate(article[0][:5]):
                                sumprobabilities[0] += snippet[0][0]
                                sumprobabilities[1] += snippet[0][1]
                            sumprobabilities[0] /= i+1
                            sumprobabilities[1] /= i+1
                            
                            #get ling feats
                            with open("../resources//dataFiles//lingFeatures//out/"+file_ , 'r') as lFile:
                                lFileDict = json.load(lFile)

                            articleFeats[2] = sumprobabilities[0]#prob support
                            articleFeats[3] = sumprobabilities[1]#prob refute
                            articleFeats.update(lFileDict[article[2]])#Use url as key in lingfeats
                            intarticleFeats = {}
                            for feat in articleFeats:
                                intarticleFeats[int(feat)] = articleFeats[feat]
                            socialData[1].append(intarticleFeats)#features
                            socialData[2].append(article[1]) #domain
                            socialData[3].append(file_)
    
    with open('../resources//pureSocialData.pickle', 'wb') as handle:
        pickle.dump(socialData, handle, protocol=pickle.HIGHEST_PROTOCOL)


def checkForSocials():
    """
        Using to check for social media portion of study
    """
    socials = ("facebook", "twitter", "quora", "reddit", "wordpress", "blogspot", "tumblr", "pinterest", "wikia")

    claimCount = 0
    for file_ in os.listdir("../resources//contentTrain//out"):
        articleCount = 0
        if file_.endswith(".json"):
            with open("../resources//contentTrain//out/" + file_, 'r') as doc:
                claimData =  json.loads(doc.read())


            for article in claimData[1]:
                for x in socials:
                    if x in article[1]:
                        articleCount += 1
            
            if articleCount > 2:
                claimCount += 1
    print(claimCount)

def checkLongTail():
    """
        shows counts of claims with certain numbers of supporting articles
    """
    claimCount = defaultdict(lambda: 0)

    for x in range(0,31):
        claimCount[x]

    for file_ in os.listdir("../resources//contentTrain//out"):
        articleCount = 0
        if file_.endswith(".json"):
            with open("../resources//contentTrain//out/" + file_, 'r') as doc:
                claimData =  json.loads(doc.read())

            
            articleCount = len(claimData[1])

        claimCount[articleCount] += 1

    for x in claimCount:
        print("articleCount: claimCount")
        print(f'{x}: {claimCount[x]}')




def getLongTail(maxArticleCount):

      # [[truthvalues], [{features}], [domain], [file_name]]
    longTailData = [[],[],[],[]]

    for file_ in os.listdir("../resources//contentTrain//out"):
        articleCount = 0
        if file_.endswith(".json"):
            with open("../resources//contentTrain//out/" + file_, 'r') as doc:
                claimData =  json.loads(doc.read())


            articleCount = len(claimData[0])
            
            if articleCount > 0 and articleCount <= maxArticleCount:
                for article in claimData[1]:
                    articleFeats = {}
                    longTailData[0].append(claimData[0][0])
                    sumprobabilities = [0,0]

                    #avg probabilities of first 5 snippets in article
                    for i, snippet in enumerate(article[0][:5]):
                        sumprobabilities[0] += snippet[0][0]
                        sumprobabilities[1] += snippet[0][1]
                    sumprobabilities[0] /= i+1
                    sumprobabilities[1] /= i+1
                    
                    #get ling feats
                    with open("../resources//dataFiles//lingFeatures//out/"+file_ , 'r') as lFile:
                        lFileDict = json.load(lFile)

                    articleFeats[2] = sumprobabilities[0]#prob refute
                    articleFeats[3] = sumprobabilities[1]#prob support
                    articleFeats.update(lFileDict[article[2]])#Use url as key in lingfeats
                    intarticleFeats = {}
                    for feat in articleFeats:
                        intarticleFeats[int(feat)] = articleFeats[feat]
                    longTailData[1].append(intarticleFeats)#features
                    longTailData[2].append(article[1]) #domain
                    longTailData[3].append(file_)
    
    return longTailData

if __name__ == "__main__":
    # checkForSocials()
    checkLongTail()
    # getSocialsDataSet()


