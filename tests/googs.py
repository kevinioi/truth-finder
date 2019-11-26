
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
import pickle
import numpy as np
from googlesearch import search

CREDIBILITY_MODEL = "../resources//models/distantSupervisionV2M2.model"
STANCE_MODEL = "../resources/models/stance2v2.model"

class article:
    stance = [-1 ,[0,0]]
    snippets = []
    lingFeats = {}
    reliability = 0

    def __init__(self, *args):
        pass        

def logDeco(func):
    def wrap(*args, **kwargs):
        print(func.__name__ + " Started...")
        results = func(*args, **kwargs)
        print(func.__name__ + " Completed...")
        return results
    return wrap

def main():
    # get user input
    print("Please Enter Your Claim:")
    claim = input(">")

    #pull top 30 google results
    results = googleQuery(claim, 30)

    #scrape all the articles
    articleData = pullWebsourceData(claim, results)

    preprocessed = preprocessData(articleData)

    credibility = predictCredibility(articleData, preprocessed)

    explanation = getExplanatorySnippet(articleData, credibility)

    print(f"Claim: {claim}")
    print("*****************************************")
    print(f"Verdict: {1 == credibility}")
    print("*****************************************")
    print("Explanation:")
    print(explanation[1])
    print(f"***\nAquired from: {explanation[2]}")

@logDeco
def googleQuery(queryString, numResults = 10):
    """
        fires a query to the google search engine, pulling url results

        return: list of search results in tuples, [(url1,domain1), (url2,domain2)...]

        param: queryString: string to query in the google search engine
        param: numResults: number of urls to be returned, default 10
    """
    results = []
    for url in search(queryString, stop=numResults):
        results.append((url, url.split('/')[2]))

    return results

@logDeco
def pullWebsourceData(claim, urlList):
    """
        Goes to each website in the urlList and pulls the releven information from the html code.
        Stores the information in 

        return: list of article objects

        param: claim: String claim to reference article to
        param: urlList: tuples (url, domain) to be scraped 
    """
    # load liguistic features
    with open('correctedLingfeats.pickle', 'rb') as handle:
        lFeatsDict =  pickle.load(handle)
    lFeatsDict = defaultdict(lambda: 0, lFeatsDict)

    # load stance Classifier and features
    stanceFeatures = featureBag.getFeatureFile("../resources/stanceFeatsV2.pickle")
    stanceClassifier = llu.load_model(STANCE_MODEL)

    # load reliabilities
    with open("../resources//compiledReliabilityDict707.txt", "r") as relFile:
        relDict = json.load(relFile)
    relDict = defaultdict(lambda: -1, relDict)

    #will hold all of the article objects
    articles = []

    for urlTuple in urlList:
        url = urlTuple[0]
        newArticle = article()
        
        try: 
            text = textProcessor.pullArticleText(url,timeoutTime=20)
            snippets = textProcessor.getSnippets(text, 4)
            releventSnips = textProcessor.getRelevence(claim,snippets)
            numRelevent = len(releventSnips[0])         

            if numRelevent > 0:
                #Getting linguistic features in article
                articleLinguisticFeatures = textProcessor.prepArticleForClassification(text, lFeatsDict)
                article.lingFeats = articleLinguisticFeatures

                #predicting stance classification
                snipData = textProcessor.prepListForClassification(releventSnips[0], stanceFeatures)
                p_labels, p_acc, p_vals = llu.predict( [], snipData, stanceClassifier, '-b 1 -q')

                #discounting snippet probs by overlap
                stanceImpact = []
                for index, probVals in enumerate(p_vals):
                    probs = [[0,0], releventSnips[0][index], url] # [[overlap*probS, overlap*prob], snippet, url]
                    probs[0][0] = (releventSnips[1])[index]*probVals[0]
                    probs[0][1] = (releventSnips[1])[index]*probVals[1]
                    stanceImpact.append(probs)
                
                #sort by max overlap*probStance
                stanceImpact.sort(key= lambda instance: max(instance[0][0], instance[0][1]),reverse=True)
                stanceImpact = stanceImpact[:7]#KEEPING 7 RELEVENT SNIPS USE 5******* 

                #attaching snippets and probabilites to article
                newArticle.snippets = stanceImpact

                #establish support or refute stance for article
                sumProbs = [0.0,0.0]
                for stance in stanceImpact[:5]:
                    sumProbs = np.add(sumProbs, stance[0])
                newArticle.stance = [int(sumProbs[0] < sumProbs[1]), sumProbs]

                # establish reliability for article based on domain
                #set to 0.5 if new domain, minimum 0.1
                newArticle.reliability = relDict[url[1]]
                if newArticle.reliability == -1:
                    newArticle.reliability = 0.5
                elif newArticle.reliability < 0.1:
                    newArticle.reliability = 0.1

                # attaching article to list of articles for claim
                articles.append(newArticle)

        except Exception as e: #
            # raise e
            continue

    return articles

@logDeco
def preprocessData(articleList):
    """
        takes stored data and converts it into vectors useable by liblinear 
    """
                    #[{features}, {features}...]
    articleFeatureList = []

    for article in articleList:
        artFeatureDict = {}
        artFeatureDict[2] = article.stance[1][0]
        artFeatureDict[3] = article.stance[1][1]
        articleFeatureList.append(artFeatureDict)

    return articleFeatureList

@logDeco
def predictCredibility(articles, articlesFeatureList):
    # load credibility model
    credModel = llu.load_model(CREDIBILITY_MODEL)

    #make article predictions
    p_labels, p_acc, p_vals = llu.predict([], articlesFeatureList, credModel, '-b 1')

    sumCurrentProbs = [0,0]
    for i, probs in enumerate(p_vals):#run through results adjusting for reliability and summarizing for full claims
        sumCurrentProbs[0] = probs[0]*(articles[i].reliability)
        sumCurrentProbs[1] = probs[1]*(articles[i].reliability)

    print(sumCurrentProbs)
    if sumCurrentProbs[0] >= sumCurrentProbs[1]:
        return 0
    else:
        return 1


def getExplanatorySnippet(articleList, claimCred):
    """
    Run through snippets grabbing the snippet which most strongly supports the determined stance of the claim

    return: snippet String 

    param: articleList: list of article objects relevent to the claim
    param: claimCred: {0,1} the determined credibility of the claim 
    """
    xSnippet = ""
    xSnippetRelevence = 0

    for article in articleList:
        for snip in article.snippets:
            if snip[0][claimCred] > xSnippetRelevence:
                xSnippetRelevence = snip[0][claimCred]
                xSnippet = snip

    return xSnippet

if __name__ == "__main__":
    main()
    pass
