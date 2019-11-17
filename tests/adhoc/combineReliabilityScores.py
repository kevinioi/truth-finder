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


for file_ in os.listdir("../resources//reliability/"):
    if file_.endswith('.txt'):
        with open("../resources//reliability/" + file_, 'r') as text:
            no = False
            for line in text:
                if no:
                    try:
                        words = line.split('\t')
                        score = json.loads(words[2])
                        (reliability[words[0]])[0] += score[0] 
                        (reliability[words[0]])[1] += score[1] 
                    except:
                        continue
                if '**' in line:
                    no = True

myList = []

for domain in reliability.items():
    myList.append(domain)

myList.sort(key=lambda x: (x[1])[0])



