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
from util import textProcessor


def makeDistantSupervisionData():
    # [[truthvalues], [{features}], [domain], [file_name]]
    fullDataSet = [[], [], [], []]
    
    for file_ in os.listdir("../resources//contentTrain//out"):
        if file_.endswith(".json"):
            with open("../resources//contentTrain//out/" + file_, 'r') as doc:
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
                
                #get ling feats
                with open("../resources//dataFiles//lingFeatures//out/"+file_ , 'r') as lFile:
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
    
    with open('properDistantSupervisionData.pickle', 'wb') as handle:
        pickle.dump(fullDataSet, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    makeDistantSupervisionData()