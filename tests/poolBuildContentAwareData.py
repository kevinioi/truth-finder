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

    #load data from all files in directory
    for file_ in os.listdir(dirAdr):    
        truthValue = None

        if file_.endswith(".json"):
            articleInfo = [[], []]#hold article information for claim
            # [[truthLabels], [ [[snippets], [probabilities], "domain", "url"]] ]

            claimLingFeats = {}
            # holds all of the linguistic features present in the articles

            with open(dirAdr + "/" + file_, 'r') as doc:
                fileData =  json.loads(doc.read())

            if fileData['Credibility'] == 'false' or fileData['Credibility'] == 'mostly false':
                truthValue = 0
            else:
                truthValue = 1

            for page in fileData["Google Results"]:#load page of google results
                for resultsDict in page.values():#load sources from google page
                    for source in resultsDict:#process each source
                        if (source["domain"] != "www.snopes.com"):
                            try:
                                text = textProcessor.pullArticleText(source["link"],timeoutTime=12)
                                snippets = textProcessor.getSnippets(text, 4)
                                releventSnips = textProcessor.getRelevence(fileData["Claim"],snippets)
                                numRelevent = len(releventSnips[0])                  

                                if numRelevent > 0:

                                    """
                                        GET LINGUISTIC FEATURES
                                    """
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

                                    articleInfo[0].append(truthValue)
                                    articleInfo[1].append([stanceImpact, source["domain"], source["link"]])

                            except Exception as e:
                                # raise e
                                continue
            

            with open("../resources//contentTrain//out/" + file_, "w") as fp:
                json.dump(articleInfo, fp)

            with open("../resources//dataFiles//lingFeatures//out/" + file_, "w") as fp:
                json.dump(claimLingFeats, fp)
    return

if __name__ == "__main__":
    
    with Pool(processes=6) as pool:
        procs = []
        procs.append(pool.apply_async(buildData,("../resources//contentTrain//one",)))
        procs.append(pool.apply_async(buildData,("../resources//contentTrain//two",)))
        procs.append(pool.apply_async(buildData,("../resources//contentTrain//three",)))
        procs.append(pool.apply_async(buildData,("../resources//contentTrain//four",)))
        procs.append(pool.apply_async(buildData,("../resources//contentTrain//five",)))
        procs.append(pool.apply_async(buildData,("../resources//contentTrain//six",)))
        #wait for each process to finish
        for proc in procs:
            proc.wait()
