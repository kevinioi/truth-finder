'''
    Truth Finder Program
'''

__author__ = "Kevin Ioi"

import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))

import json
from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from util import featureBag
from util import textProcessor
from collections import defaultdict
import pickle
import numpy as np
from googlesearch import search
from pathlib import Path
from tqdm imort tqdm

class article:
    """
        object to hold article data
    """
    stance = [-1 ,[0,0]]
    snippets = []
    lingFeats = {}
    reliability = 0

    def __init__(self, *args, **kwargs):
        pass        

def logDeco(func):
    """
        a very uneccessary logging function
    """
    def wrap(*args, **kwargs):
        print(func.__name__ + " Started...")
        results = func(*args, **kwargs)
        print(func.__name__ + " Completed...")
        return results
    return wrap

def main(args):
    # get user input
    print("Please Enter Your Claim:")
    claim = input(">")

    #pull top 30 google results
    results = googleQuery(claim, 30)

    articleData = pullWebsourceData(claim, results, args)

    preprocessed = preprocessData(articleData)

    credibility = predictCredibility(articleData, preprocessed, Path(args.credModel))

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

        params:
            queryString (str):
                string to query in the google search engine
            numResults (int):
                number of urls to be returned, default 10

        return: list of search results in tuples, [(url1,domain1), (url2,domain2)...]
    """
    results = []
    for url in search(queryString, stop=numResults):
        results.append((url, url.split('/')[2]))

    return results

@logDeco
def pullWebsourceData(claim, urlList, args):
    """
        Goes to each website in the urlList and pulls the releven information from the html code.
        Stores the information in 

        params:
            claim (str):
                String claim to reference article to
            urlList (tuple):
                tuples (url, domain) to be scraped
            args (Namespace):
                contains paths to saved models and stored data
        return:
            list of article objects
    """

    # load liguistic features
    with open(Path(args.lingFeats), 'rb') as handle:
        lFeatsDict =  pickle.load(handle)
    lFeatsDict = defaultdict(lambda: 0, lFeatsDict)

    # load stance Classifier and features
    stanceFeatures = featureBag.getFeatureFile(Path(args.stanceFeats))
    stanceClassifier = llu.load_model(Path(args.stanceModel))

    # load reliabilities
    with open(Path(args.srcRel), "r") as relFile:
        relDict = json.load(relFile)
    relDict = defaultdict(lambda: -1, relDict)

    #will hold all of the article objects
    articles = []

    for urlTuple in tqdm(urlList, position=0, leave=True): # attempt to process all provided urls
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
            # will error for webserver timeouts, 
            continue

    return articles

@logDeco
def preprocessData(articleList):
    """
        takes stored data and converts it into vectors useable by liblinear

        params:
            articleList (list, articles):
                list of articles objects filled with information

        returns:
            data read to be used in liblinear package's training or prediction

    """
                    #[{features}, {features}...]
    articleFeatureList = []
    normDict = defaultdict(lambda : 0)

    for article in articleList:
        artFeatureDict = {}
        artFeatureDict[2] = article.stance[1][0]
        artFeatureDict[3] = article.stance[1][1]
        for key in artFeatureDict:
            if artFeatureDict[key] > normDict[key]:
                normDict[key] = artFeatureDict[key]
        articleFeatureList.append(artFeatureDict)

    for article in articleFeatureList:
        for feature in normDict:
            if feature in article:
                article[feature] /= normDict[feature]

    return articleFeatureList

@logDeco
def predictCredibility(articles, articlesFeatureList, credModel):
	'''
		Utilize the trained credibility model to assess the 'trustworthyness' of a claim
		based on the evidence gathered from the internet

		params:
			articles

			articlesFeatureList


		return:
			boolean
				True if credible, False if incredible 
	'''

    # load credibility model
    credModel = llu.load_model(credModel)

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

    params: 
        articleList (list, article):
            list of article objects relevent to the claim
        claimCred (int):
            {0,1} the determined credibility of the claim 
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
    parser = argparse.ArgumentParser(description='Process input parameters')
    parser.add_argument('--credModel', metavar='--credModel', type=str,
                help='path to credibility model')
    parser.add_argument('--lingFeats', metavar='--lingFeats', type=str,
                help='path to file in which store the linguistic features of articles')
    parser.add_argument('--stanceModel', metavar='--stanceModel', type=str,
            help='path to file in which store the stance model')
    parser.add_argument('--stanceFeats', metavar='--stanceFeats', type=str,
                help='path to file in which store the stance features')
    parser.add_argument('--srcRel', metavar='--srcRel', type=str,
                help='path to file in which store the reliability information')

    args = parser.parse_args()

    main(args)
