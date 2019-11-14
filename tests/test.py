import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from util import featureBag



for file_ in os.listdir("../resources//contentTrain//output"):
    if file_.endswith(".json"):
        with open("../resources//contentTrain//output/" + file_, 'r') as doc:#read snopes file
            fileData =  json.loads(doc.read())

        print(fileData[0])

    break            

# featDict = featureBag.getFeatureFile("../resources/featsV2.pickle")

