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
from collections import defaultdict
from util import featureBag
from util import textProcessor

model1 = llu.load_model("../resources//stance_models/model.liblin")
text = textProcessor.pullArticleText("https://www.truthorfiction.com/41-lb-rat-captured-in-new-york-city/")
claim = "A photograph shows a 41 lb rat that was caught in New York City."
textSnips = textProcessor.getSnippets(text, 4)
featDict = featureBag.getFeatureFile("../resources/feats.pickle")

relevent = set()

for snippet in textSnips:
    if textProcessor.calcOverlap(claim, snippet) >= 0.4:
        relevent.add(snippet)

for r in relevent:
    print("**********************************")
    print(r)
    p_labels, p_acc, p_vals = llu.predict( [], textProcessor.prepTextForClassification(r,featDict),model1, '-b 1')
    