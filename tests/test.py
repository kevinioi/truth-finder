import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

import json
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from util import featureBag
from util import textProcessor
from collections import defaultdict
import pickle
from util import textProcessor
import bs4
#dictionary containing all possible features 
# featDict = featureBag.getFeatureFile("../resources/featsV2.pickle")


s = [[1,2], [1,2], [1,2]]


# text = textProcessor.pullArticleText("https://www.snopes.com/fact-check/meijer-coupon-scam/")

# print(text)