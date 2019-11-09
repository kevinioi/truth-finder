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
from collections import defaultdict
import pickle

with open("../resources//linguisticLexicon/negative-words.txt", "r") as file_:
    content = file_.readlines()

dd = defaultdict(lambda : 0)

count=0

for i,line in enumerate(content):
    # words = line.split(" ",5)
    if dd[line.strip()] == 0:
        count+=1
        dd[line.strip()] = count

print(dd)

with open('lingFeats3.pickle', 'wb') as handle:
    pickle.dump(dict(dd), handle, protocol=pickle.HIGHEST_PROTOCOL)
   




# """
#     count refute and support
# """

# true = 0
# false = 0

# for file_ in os.listdir("../resources//snopesData"):
#     if file_.endswith(".json"):
#         with open("../resources//snopesData/" + file_, 'r') as doc:
#             fileData =  json.loads(doc.read())

#         if fileData['Credibility'] == 'false' or fileData['Credibility'] == 'mostly false':
#             false +=1
#         elif fileData['Credibility'] == 'true' or fileData['Credibility'] == 'mostly true':
#             true += 1

# print(true)
# print(false)
# print(false/true)