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
    for i in range(1,30):
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
    for i in range(1,30):
        for opinion in strengths[1][i]:
            if opinion[0] > opinion[1]:
                refuteCount += 1
                refuteSum += opinion[0]
            else:
                supportCount += 1
                supportSum += opinion[1]
        print(f"day {i}: support {supportSum}/{supportCount}  refute {refuteSum}/{refuteCount}")

    return


def getRollingAvgStance(days):
    supportCount = 0
    refuteCount = 0
    supportSum = 0
    refuteSum = 0

    rollingStanceAvg = {}

    for i in range(1,30):
        rollingStanceAvg[i] = (0,0)
        for opinion in days[i]:
            if opinion[0] > opinion[1]:
                refuteCount += 1
                refuteSum += opinion[0]
            else:
                supportCount += 1
                supportSum += opinion[1]

        #put avg stance calc in try catch to avoid divide by zero
        try:
            supportStance = supportSum/supportCount
        except:
            supportStance = 0
        try:
            refuteStance = refuteSum/refuteCount
        except:
            refuteStance = 0

        rollingStanceAvg[i] = (refuteStance,supportStance)
    
    return rollingStanceAvg


def getSlopeToDay(currentDay, days):

    rollingStanceAvg = getRollingAvgStance(days)

    #           [refuteProb, supportProb] 
    numeratorOne = [0,0]
    numeratorTwo = [0,0]
    denominatorOne = [0,0]
    denominatorTwo = [0,0]

    for t in range(1,currentDay+1):
        numeratorOne[0] += rollingStanceAvg[t][0]*t
        numeratorTwo[0] += rollingStanceAvg[t][0]
        numeratorOne[1] += rollingStanceAvg[t][1]*t
        numeratorTwo[1] += rollingStanceAvg[t][1]
        denominatorOne[0] += t**2  
        denominatorOne[1] += t**2
        denominatorTwo[0] += t
        denominatorTwo[1] += t

    numeratorOne[0] *= t
    numeratorOne[1] *= t
    numeratorTwo[0] *= (currentDay*(currentDay+1)/2)
    numeratorTwo[1] *= (currentDay*(currentDay+1)/2)
    denominatorOne[0] *= t
    denominatorOne[1] *= t
    denominatorTwo[0] = denominatorTwo[0]**2
    denominatorTwo[1] = denominatorTwo[1]**2

    slopes = [0,0]
    try:
        slopes[0] = (numeratorOne[0] - numeratorTwo[0])/(denominatorOne[0]-denominatorTwo[0])
    except:
        slopes[0] = 0
    try:
        slopes[1] = (numeratorOne[1] - numeratorTwo[1])/(denominatorOne[1]-denominatorTwo[1])
    except:
        slopes[1] = 0
    return slopes


def trendBasedCredibility(currentDay, days):
    rollingStanceAvg = getRollingAvgStance(days)

    slopes = getSlopeToDay(currentDay, days)
    cred = (1+slopes[1])*rollingStanceAvg[currentDay][1] - (1+slopes[0])*rollingStanceAvg[currentDay][0]

    return cred

if __name__ == "__main__":

    # sumStrenthOverTime()

    with open("../resources//timeSeries//out/fullData.json", "r") as fp:
        fullData = json.load(fp)

    #           [false strength, true strength]
    strengths = [{}, {}]

    for i in range(1,30):
        strengths[0][i] = []
        strengths[1][i] = []


    for claim in fullData:
        for day in fullData[claim][0]:
            for article in fullData[claim][0][day]:#for each article append the reliability corrected stance probabilites
                strengths[fullData[claim][1]][int(day)].append((article[0]['2']*article[0]['1'],article[0]['3'] * article[0]['1']))

    for day in range(1,30):
        x = trendBasedCredibility(day, strengths[1])
        # x = trendBasedCredibility(day, strengths[0])#refute class
        print(x)
    