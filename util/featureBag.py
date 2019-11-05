import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
import json
import nltk
from nltk.tokenize import word_tokenize
import pickle
from collections import defaultdict

def defDictFunc():
    """
        Used to set the default value of dictionary
        Pickle requires it to be a function not a lambda

        effectively returns 0 
    """
    return int

def getFeatureFile(sourceAdr):
    """
        Loads the previously created features file

        returns dict
    """
    try:
        with open(sourceAdr, 'rb') as handle:
            myDict = pickle.load(handle)
    except Exception as e:
        raise e

    return myDict

def createFeatureFile(sourceDir, outputAdr):
    """
        Contruct the save file containing all 'features'
    """

    features = defaultdict(defDictFunc())

    count = 0

    #load data from all source files
    for file_ in os.listdir(sourceDir):
        if file_.endswith(".json"):
            with open(sourceDir + file_, 'r') as doc:
                fileData =  json.loads(doc.read())

            #grab description and chunk the data
            wordBag = word_tokenize(fileData['Description'])
            wordBagTagged = nltk.pos_tag(wordBag)
            wordBag = [word[0] for word in wordBagTagged if ((word[1] != 'NNP') & (word[1] != 'NNPS'))]
            wordBag = [word.lower() for word in wordBag if word.isalpha()]

            #get uni/bigrams
            unigrams = nltk.ngrams(wordBag,n=1)
            bigrams = nltk.ngrams(wordBag,n=2)
            
            #add data to features dictionary
            for gram in unigrams:
                if gram not in features:
                    count +=1
                    features[gram] = count
            for gram in bigrams:
                if gram not in features:
                    count +=1
                    features[gram] = count















    #Save to file
    with open(outputAdr, 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

