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
import requests
import eventlet

# mydict = {'www.mlive.com': [1, 0], 'www.wilx.com': [1, 0], 'www.freep.com': [1, 0], 'woodtv.com': [1, 0], 'meijermadness.com': [0, 1], 'www.bargainstobounty.com': [1, 0], 'www.cheapassgamer.com': [1, 0], 'www.prnewswire.com': [0, 1], 'www.onlinethreatalerts.com':[1, 0], 'www.amittenfullofsavings.com': [0, 1], 'www.bargainist.com': [1, 0], 'bargainbriana.com': [0, 1], 'browst.com': [1, 0], 'stephaniesavings.wordpress.com': [0, 1], 'www.tumblr.com': [1, 0], 'slickdeals.net': [0, 1]}

# for i, text in enumerate(mydict):
#     print(i, text, mydict[text])

# text = textProcessor.pullArticleText("https://www.snopes.com/fact-check/meijer-coupon-scam/")
# text = textProcessor.pullArticleText("http://portal.clubrunner.ca/2010/Stories")

# print(len(text))

# for file_ in os.listdir("../resources//partialSnopes"):    
#     print(file_)

test = []
test.append([1,11,3])
test.append([4,2,3])
test.append([2,2,9])
test.append([3,6,3])

test.sort(key= lambda instance: max(instance[1], instance[2]),reverse=True)

print(test)


