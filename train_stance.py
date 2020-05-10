'''
    Module used to train the stance classification model 
'''

__author__ = "Kevin Ioi"
__date__ = "Nov_2019"

# system imports
import os 
import sys
from pathlib import Path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

# data in/out imports
import json
import pickle
import argparse

import numpy as np
from sklearn import metrics
from nltk import pos_tag, ngrams
from nltk.tokenize import word_tokenize
from collections import defaultdict
from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from tqdm import tqdm
from util import featureBag


def logger()

def main(args):

    featureDict = featureBag.createFeatureFile(Path(args.dataPath), Path(args.featPath))
    train_model(featureDict, Path(args.dataPath), Path(args.stancePath)) 
    print("Done!")

@logger
def train_model(featDict, dataPath, modelPath):
    '''
        Trains the stance determination model

        params:
            featDict (os Path):
                path to address in which to save features for stance
            dataPath (os Path):
                path to directoy that contains all the training data
            modelPath (os Path):
                path to address in which to save stance determination model
    '''

    dataSet = ([], [])
    truthValue = None

    #load data from all source files
    for file_ in tqdm(os.listdir(dataPath), position=0, leave=True):
        if file_.endswith(".json"):
            with open(os.path.join(dataPath,file_), 'r') as doc:
                fileData =  json.loads(doc.read())

            if fileData['Credibility'] == 'false' or fileData['Credibility'] == 'mostly false':
                truthValue = 0
            elif fileData['Credibility'] == 'true' or fileData['Credibility'] == 'mostly true':
                truthValue = 1

            wordBag = word_tokenize(fileData['Description'])
            wordBagTagged = pos_tag(wordBag)
            wordBag = [word[0] for word in wordBagTagged if ((word[1] != 'NNP') & (word[1] != 'NNPS'))]
            wordBag = [word.lower() for word in wordBag if word.isalpha()]

            #get uni/bigrams
            unigrams = ngrams(wordBag,n=1)
            bigrams = ngrams(wordBag,n=2)
            
            #add data to training/testing Sample
            dataSet[0].append(truthValue)
            features = defaultdict(featureBag.defDictFunc())

            for gram in unigrams:
                if featDict[gram] != 0:#don't count gram that aren't known features
                    features[featDict[gram]] += 1
            for gram in bigrams:
                if featDict[gram] != 0:#don't count gram that aren't known features
                    features[featDict[gram]] += 1
            dataSet[1].append(dict(features))

    model = llu.train(dataSet[0], dataSet[1], '-s 0 -c 4 -w1 2.7 -v 10')
    llu.save_model(modelPath,model)


def evauluateModel(dataSet, model):
    '''
        evaluates the accuracy of the model 
    '''

    p_labels, p_acc, p_vals = llu.predict(dataSet[0], dataSet[1], model, '-b 1')

    correctTrue = 0
    correctFalse = 0
    wrongTrue = 0
    wrongFalse = 0

    probs = []
    for i, x in enumerate(p_labels):
        probs.append(p_vals[i][1])
        if int(x) != int(dataSet[0][i]):
            if int(dataSet[0][i]) == 1:
                wrongTrue += 1
            else:
                wrongFalse += 1
        else:
            if int(dataSet[0][i]) == 1:
                correctTrue += 1
            else:
                correctFalse += 1
    
    #ROC/AUC calculations
    fpr, tpr, thresholds = metrics.roc_curve(dataSet[0], probs)
    auc = metrics.auc(fpr, tpr)

    print(f'AUC = {auc}')
    print(f"wrongFalse = {wrongFalse}")
    print(f"wrongTrue = {wrongTrue}")

    return [(wrongFalse, wrongTrue), (correctFalse, correctTrue)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input parameters')
    parser.add_argument('--dataPath', metavar='--dataPath', type=str,
                help='path to directory containing data files')
    parser.add_argument('--featPath', metavar='--featPath', type=str,
                help='path to file in which store the stance features')
    parser.add_argument('--stancePath', metavar='--stancePath', type=str,
            help='path to file in which store the stance model')

    args = parser.parse_args()

    main(args)
        