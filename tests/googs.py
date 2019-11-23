
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

from googlesearch import search

class article:
    stance = 0
    snips = []
    lingFeats = {}
    reliability = 0

    def __init__(self, *args):
        pass        


# load stance Classifier and features
features = featureBag.getFeatureFile("../resources/stanceFeatsV2.pickle")
stanceClassifier = llu.load_model("../resources/models/stance2v2.model")

# load credibility model
credModel = llu.load_model("../resources//models/jointModelSample.model")

# load liguistic features
with open('correctedLingfeats.pickle', 'rb') as handle:
    lFeatsDict =  pickle.load(handle)
lFeatsDict = defaultdict(lambda: 0, linguisticFeatures)

# load reliabilities
with open("../resources//compiledReliabilityDict702.txt", "r") as relFile:
    relDict = json.load(relFile)
relDict = defaultdict(lambda: -1, relDict)


def logDeco(func):
    def wrap(*args, **kwargs):
        print(func.__name__ + " Started...")
        results = func(*args, **kwargs)
        print(func.__name__ + " Completed...")
        return results
    return wrap


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

        param: urlList: tuples (url, domain) to be scraped 
    """
    articles = []

    for url in urlList:
        newArticle = article()
        
        try: 
            text = textProcessor.pullArticleText(url,timeoutTime=12)
            snippets = textProcessor.getSnippets(text, 4)
            releventSnips = textProcessor.getRelevence(claim,snippets)
            numRelevent = len(releventSnips[0])         

            if numRelevent > 0:
                lingFeats = textProcessor.prepArticleForClassification(text, linguisticFeatures)
                claimLingFeats[source["link"]] = lingFeats
                """
                """

                snipData = textProcessor.prepListForClassification(releventSnips[0],features)
                p_labels, p_acc, p_vals = llu.predict( [], snipData, model, '-b 1 -q')

                stanceImpact = []
                for index, probVals in enumerate(p_vals):
                    probs = [[0,0], releventSnips[0][index]] # [[overlap*probS, overlap*prob], snippet]
                    probs[0][0] = (releventSnips[1])[index]*probVals[0]
                    probs[0][1] = (releventSnips[1])[index]*probVals[1]
                    stanceImpact.append(probs)
                
                #sort by max overlap*probStance
                stanceImpact.sort(key= lambda instance: max(instance[0][0], instance[0][1]),reverse=True)

                stanceImpact = stanceImpact[:7]#KEEPING 7 RELEVENT SNIPS USE 5******* 

                newArticle.snippets = stanceImpact
                newArticle.stance = 

                articleInfo[0].append(truthValue)
                articleInfo[1].append([stanceImpact, source["domain"], source["link"]])
        except: #Exception as e
            # raise e
            continue
    
    
    
    return None

@logDeco
def preprocessData():
    """
        takes stored data and converts it into vectors useable by liblinear 
    """


    return None


if __name__ == "__main__":
    # get user input
    print("Please Enter Your Claim:")
    claim = input(">")

    #pull top 30 google results
    results = googleQuery(claim, 30)

    pullWebsourceData(claim, results)


    

    pass