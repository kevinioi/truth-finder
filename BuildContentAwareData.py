'''
    Module used to webscrape and compile data into a format usable
    to train the credibility classifier model

    Before running you MUST have:
    1. Trained the Stance Classifier model
    2. Downloaded the Snopes dataset and placed it the resources directory

'''

import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

from util import featureBag
from util import textProcessor
from collections import defaultdict

import multiprocessing
from multiprocessing import Pool
import numpy as np
import pickle
import json


SNOPES_DATA_DIR = os.path.join("resources","Snopes")

STANCE_FEATS = os.path.join("resources","stanceFeats.pkl")
STANCE_MODEL = os.path.join("resources","models","stance.model")

RAW_CRED_DATA_OUT = os.path.join("resources","webdata_raw")
CRED_MODEL_DATA = os.path.join('resources","distantSupervisionData.pkl')

LING_FEATS_RAW = os.path.join("resources","lingfeats_raw")
LING_FEATS = os.path.join("resources","lingfeats.pkl")

RAW_REL_OUT = os.path.join("resources","reliability_raw")
COMPILED_REL = os.path.join("resources","reliabilityDict.json")

REQUIRED_DIRS = ["resources",
                os.path.join("resources", "models"),
                LING_FEATS_RAW,
                RAW_CRED_DATA_OUT,
                RAW_REL_OUT]

def main():
	processor_count = multiprocessing.cpu_count()
    dataFiles = os.listdir(os.path.join("resources","Snopes"))

    fileSplits = np.array_split(dataFiles, processor_count)

    with Pool(processes=processor_count) as pool:
        procs = []

        for fileList in fileSplits:
        	procs.append(pool.apply_async(webScrapeData,(fileList,)))
        #wait for each process to finish
        for proc in procs:
            proc.wait()

    combineReliabilityScores()
    compileDistantSupervisionData(RAW_CRED_DATA_OUT)

def webScrapeData(fileList):
    features = featureBag.getFeatureFile(STANCE_FEATS)
    model = llu.load_model(STANCE_MODEL)

    with open(LING_FEATS, 'rb') as handle:
        linguisticFeatures =  pickle.load(handle)
    linguisticFeatures = defaultdict(lambda: 0, linguisticFeatures)

    #load data from all files in directory
    for file_ in fileList:    
        truthValue = None

        if file_.endswith(".json"):
            articleInfo = [[], []]#hold article information for claim
            # [[truthLabels], [ [[snippets], [probabilities], "domain", "url"]] ]

            claimLingFeats = {} # holds all of the linguistic features present in the articles

            reliability = defaultdict(lambda : [0,0])# holds reliability scores of the domains with relevent articles

            with open(os.path.join(SNOPES_DATA_DIR, file_), 'r') as doc:
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

                                    lingFeats = textProcessor.prepArticleForClassification(text, linguisticFeatures)
                                    claimLingFeats[source["link"]] = lingFeats

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

                                    probSum = [0,0]
                                    stanceImpact = stanceImpact[:6]
                                    for index, probVals in enumerate(stanceImpact):
                                        probSum[0] += probVals[0][0]
                                        probSum[1] += probVals[0][1]
                                    probSum[0] /= index + 1
                                    probSum[1] /= index + 1

                                    if (probSum[truthValue] > probSum[abs(truthValue-1)]):
                                        (reliability[source["domain"]])[0] += 1#correct
                                    else:
                                        (reliability[source["domain"]])[1] += 1#incorrect

                                    articleInfo[0].append(truthValue)
                                    articleInfo[1].append([stanceImpact, source["domain"], source["link"]])

                            except:
                            	# catch exceptions related to webpage timeout, move on  
                                continue

            #save file reliability scores
            with open(os.path.join(RAW_REL_OUT,file_), "w") as fp:
                for r in reliability:
                    articleStances = reliability[r]
                    percentCorrect = articleStances[0]/(articleStances[0]+ articleStances[1])
                    fp.write(r + "\t" + str(percentCorrect) + "\t" + str(articleStances) + "\n")

            #save file features, probabilities and snippets
            with open(os.path.join(RAW_CRED_DATA_OUT,file_), "w") as fp:
                json.dump(articleInfo, fp)

            #save file linguistic features
            with open(os.path.join(LING_FEATS_RAW,file_), "w") as fp:
                json.dump(claimLingFeats, fp)

def compileDistantSupervisionData(infoAdr):
    """ 
        formats raw data in to a usable shape for the training of 
        the distant supervision based credibility model

        params:
            infoAdr (os path): 
                path to the directory containing the webscraped data
    """
    # [[truthvalues], [{features}], [domain], [file_name]]
    fullDataSet = [[], [], [], []]

    for file_ in os.listdir(infoAdr):
        if file_.endswith(".json"):
            with open(os.path.out(infoAdr,file_), 'r') as doc:
                claimData =  json.loads(doc.read())
            
            for article in claimData[1]:
                articleFeats = {}
                fullDataSet[0].append(claimData[0][0])#truth value
                sumprobabilities = [0,0]

                #avg probabilities of first 5 snippets in article
                for i, snippet in enumerate(article[0][:5]):
                    sumprobabilities[0] += snippet[0][0]
                    sumprobabilities[1] += snippet[0][1]
                sumprobabilities[0] /= i+1
                sumprobabilities[1] /= i+1
                
                #get ling feats for specific articles
                with open(os.path.join(LING_FEATS,file_), 'r') as lFile:
                    lFileDict = json.load(lFile)

                articleFeats[2] = sumprobabilities[0]#prob support
                articleFeats[3] = sumprobabilities[1]#prob refute
                articleFeats.update(lFileDict[article[2]])#Use url as key in lingfeats
                intarticleFeats = {}
                for feat in articleFeats:
                    intarticleFeats[int(feat)] = articleFeats[feat]
                fullDataSet[1].append(intarticleFeats)#features
                fullDataSet[2].append(article[1]) #domain
                fullDataSet[3].append(file_)
    
    with open(CRED_MODEL_DATA, 'wb') as fp:
        pickle.dump(fullDataSet, fp, protocol=pickle.HIGHEST_PROTOCOL)


def combineReliabilityScores():
    """ Compile reliability scores """

    reliability = defaultdict(lambda: [0,0])

    for file_ in os.listdir(RAW_REL_OUT):
        if file_.endswith('.json'):
            with open(os.path.join(RAW_REL_OUT,file_), 'r') as fileText:
                for line in fileText:
                    lineText = line.split('\t')
                    score = json.loads(words[2])
                    domain = lineText[0]
                    reliability[domain] = np.add(reliability[domain], score)

    domain_reliability_rating = {}

    sortme = []
    for x in reliability:
        domain_reliability_rating[x] = reliability[x][0]/(reliability[x][0]+reliability[x][1]) 

    with open(COMPILED_REL, "w") as fp:
        json.dump(domain_reliability_rating, fp)


if __name__ == "__main__":

	if not os.path.exists(os.path.join("resources/Snopes/")):
		raise Exception("resources/Snopes/\nSnopes data cannot be found. Please ensure you have downloaded and \
						unpacked the snopes data in the the resources directory")

    for directory in REQUIRED_DIRS:
        if not os.path.exists(directory):
            os.mkdir(directory)

	main()