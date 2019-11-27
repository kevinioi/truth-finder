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
                    doit =True

                    for x in socials:
                        if x in article[1]:
                            doit = False

                    if doit == True:
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
    
    with open('../resources//pureWebSocialData.pickle', 'wb') as handle:
        pickle.dump(socialData, handle, protocol=pickle.HIGHEST_PROTOCOL)


def getJointSocialDataSet():
    """
        CHANGE THE SAVE FILE BEFORE YOU RUN IT
        
        make sure these files are up to date:
        - features/stances of articles is up to date --->    ../resources//contentTrain//out
        - linguistic features are up to date --->     ../resources//dataFiles//lingFeatures//out/
    """

    # [[truthvalues], [{features}], [file_name]]
    socialData = [[], [], []]

    socials = ("facebook", "twitter", "quora", "reddit", "wordpress", "blogspot", "tumblr", "pinterest", "wikia")

    #reliability data
    # with open("../resources//compiledReliabilityDict707.txt", "r") as relFile:
    with open("../resources//compiledReliabilityDictFINAL.txt", "r") as relFile:
        relDict = json.load(relFile)
    relDict = defaultdict(lambda: -1, relDict)

    # for file_ in os.listdir("../resources//wikiData//outPeople"):
    for file_ in os.listdir("../resources//contentTrain//out"):
        if file_.endswith(".json"):
            articleCount = 0


            #features = 1:reliability, 2:prob support?, 3: prob refute?, 4+: lingFeats
            # fullClaimData = [truth value,{features},file_]
            fullClaimData = [-1,defaultdict(lambda: 0),""]

            # with open("../resources//wikiData//outPeople/" + file_, 'r') as doc:
            with open("../resources//contentTrain//out/" + file_, 'r') as doc:
                claimData =  json.loads(doc.read())

            for article in claimData[1]:
                for x in socials:
                    if x in article[1]:
                        articleCount += 1
            if articleCount > 2:

                if len(claimData[0]) > 0:
                    fullClaimData[0]= (claimData[0])[0]#truth value
                    fullClaimData[2] = file_

                    for article in claimData[1]:
                        ar = True

                        for x in socials:
                            if x in article[1]:
                                ar = False

                        if ar == True:
                            articleFeats = {}
                            sumprobabilities = [0,0]

                            #avg probabilities of first 5 snippets in article
                            for i, snippet in enumerate(article[0][:5]):
                                sumprobabilities[0] += snippet[0][0]
                                sumprobabilities[1] += snippet[0][1]
                            sumprobabilities[0] /= i+1
                            sumprobabilities[1] /= i+1
                            
                            #get ling feats
                            # with open("../resources//wikiData//lingFout/"+file_ , 'r') as lFile:
                            with open("../resources//dataFiles//lingFeatures//out/"+file_ , 'r') as lFile:
                                lFileDict = json.load(lFile)

                            articleFeats[2] = sumprobabilities[0]#prob refute
                            articleFeats[3] = sumprobabilities[1]#prob support
                            articleFeats.update(lFileDict[article[2]])#Use url as key in lingfeats
                            for feat in articleFeats:
                                fullClaimData[1][int(feat)] += articleFeats[feat] #add features to dictionary representing aggregate claim features

                            #get reliability
                            reliability = relDict[article[1]]
                            if reliability == -1:#new domian
                                reliability = 0.5
                            elif reliability < 0.25:#baseline reliability
                                reliability = 0.25
                            fullClaimData[1][1] += reliability

                    socialData[0].append(fullClaimData[0])#truth value
                    socialData[1].append(dict(fullClaimData[1])) #feature dict
                    socialData[2].append(fullClaimData[2])#file name

    with open('../resources//socialWebJointData.pickle', 'wb') as handle:
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
    # checkLongTail()
    getSocialsDataSet()
    
    # getJointSocialDataSet()

