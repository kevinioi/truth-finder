import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

import json
from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from util import featureBag
import pickle
from collections import defaultdict


def sumStrenthOverTime():

    #           [false strength, true strength]
    strengths = [{}, {}]

    for i in range(1,21):
        strengths[0][i] = []
        strengths[1][i] = []
    
    with open("../resources//timeSeries//out/fullData.json", "r") as fp:
        fullData = json.load(fp)

    for claim in fullData:
        for day in fullData[claim][0]:
            for article in fullData[claim][0][day]:#for each article append the reliability corrected stance probabilites
                strengths[fullData[claim][1]][int(day)].append((article[0]['2']*article[0]['1'],article[0]['3'] * article[0]['1']))

    #sum and print rolling probabilities
    supportCount = 0
    refuteCount = 0
    supportSum = 0
    refuteSum = 0
    for i in range(1,21):
        for opinion in strengths[0][i]:
            if opinion[0] > opinion[1]:
                refuteCount += 1
                refuteSum += opinion[0]
            else:
                supportCount += 1
                supportSum += opinion[1]
        print(f"day {i}: support {supportSum/supportCount}  refute {refuteSum/refuteCount}")

    print("**********************************************")
    supportCount = 0
    refuteCount = 0
    supportSum = 0
    refuteSum = 0
    for i in range(1,21):
        for opinion in strengths[1][i]:
            if opinion[0] > opinion[1]:
                refuteCount += 1
                refuteSum += opinion[0]
            else:
                supportCount += 1
                supportSum += opinion[1]
        print(f"day {i}: support {supportSum}/{supportCount}  refute {refuteSum}/{refuteCount}")

    return


if __name__ == "__main__":
    sumStrenthOverTime()