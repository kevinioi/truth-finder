import sys
import os
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


def checkForSocials():
    """
        Using to check for social media portion of study
    """
    socials = ("facebook", "twitter", "quora", "reddit", "wordpress", "blogspot", "tumblr", "pinterest", "wikia")

    claimCount = 0
    for file_ in os.listdir("../resources//contentTrain//out"):
        articleCount = 0
        if file_.endswith(".json"):
            with open("../resources//contentTrain//out/" + file_, 'r') as doc:
                claimData =  json.loads(doc.read())


            for article in claimData[1]:
                for x in socials:
                    if x in article[1]:
                        articleCount += 1
            
            if articleCount > 2:
                claimCount += 1
    print(claimCount)

def checkLongTail():

    claimCount = defaultdict(lambda: 0)

    for file_ in os.listdir("../resources//contentTrain//out"):
        articleCount = 0
        if file_.endswith(".json"):
            with open("../resources//contentTrain//out/" + file_, 'r') as doc:
                claimData =  json.loads(doc.read())

            for _ in claimData[1]:
                articleCount+=1

        claimCount[articleCount] += 1

    for x in claimCount:
        print("articleCount: claimCount")
        print(f'{x}: {claimCount[x]}')


if __name__ == "__main__":
    # checkForSocials()
    checkLongTail()


