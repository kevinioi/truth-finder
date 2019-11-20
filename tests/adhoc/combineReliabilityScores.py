#!/usr/bin/env python3

import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from util import featureBag

import json
from collections import defaultdict

"""
    Compile reliability scores
"""

reliability = defaultdict(lambda: [0,0])


for file_ in os.listdir("../resources//reliability//output"):
    if file_.endswith('.json'):
        with open("../resources//reliability//output/" + file_, 'r') as text:
            # no = False
            for line in text:
                # if no:
                try:
                    words = line.split('\t')
                    score = json.loads(words[2])
                    (reliability[words[0]])[0] += score[0] 
                    (reliability[words[0]])[1] += score[1] 
                except Exception as e:
                    raise e
                # if '**' in line:
                    # no = True

completeDict = {}

for x in reliability:
    completeDict[x] = reliability[x][0]/(reliability[x][0]+reliability[x][1]) 

# for domain in reliability.items():
#     if ((domain[1])[0] +(domain[1])[1]) >= 3:
#         myList.append(domain)

# myList.sort(key=lambda x: ((x[1])[0]*1.0)/((x[1])[0] +(x[1])[1]))




with open("compiledReliabilityDict502.txt", "w") as a:
    a.write(json.dumps(completeDict))

    



