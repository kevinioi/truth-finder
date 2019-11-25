import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

import json
from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from util import featureBag
from util import textProcessor
from collections import defaultdict
from multiprocessing import Pool



"""
    takes raw snopes files from ../resources//reliability//one, two, three
    - Pulls reliability of individual articles for each claim

    Stores the information under filename in ../resources//reliability//out/


    ****RUN combineReliabilityScores after determineRel()
"""


def determineRel(dirAdr):
    """
        pull from web to determine source reliabilities
    """
    features = featureBag.getFeatureFile("../resources/stanceFeatsV2.pickle")
    model = llu.load_model("../resources/models/stance2v2.model")

    #load data from all source files
    for file_ in os.listdir(dirAdr):    
        truthValue = None

        if file_.endswith(".json"):
            reliability = defaultdict(lambda : [0,0])

            with open(dirAdr + "/" + file_, 'r') as doc:
                fileData =  json.loads(doc.read())

            if fileData['Credibility'] == 'false' or fileData['Credibility'] == 'mostly false':
                truthValue = 0
            else:
                truthValue = 1

            for page in fileData["Google Results"]:#load page of google results
                for resultsDict in page.values():#load sources from google page
                    for source in resultsDict:#process each source
                        if (source["domain"] != "www.snopes.com"):
                            try:
                                text = textProcessor.pullArticleText(source["link"],timeoutTime=10)
                                snippets = textProcessor.getSnippets(text, 4)
                                releventSnips = textProcessor.getRelevence(fileData["Claim"],snippets)
                                numRelevent = len(releventSnips[0])                  

                                if numRelevent > 0:
                                    snipData = textProcessor.prepListForClassification(releventSnips[0],features)
                                    p_labels, p_acc, p_vals = llu.predict( [], snipData, model, '-b 1 -q')

                                    stanceImpact = []
                                    for index, probVals in enumerate(p_vals):
                                        probs = [0,0]
                                        probs[0] = (releventSnips[1])[index]*probVals[0]
                                        probs[1] = (releventSnips[1])[index]*probVals[1]
                                        stanceImpact.append(probs)
                                    stanceImpact.sort(key= lambda instance: max(instance[0], instance[1]),reverse=True)

                                    probSum = [0,0]
                                    for index, probVals in enumerate(stanceImpact[:6]):
                                        probSum[0] += probVals[0]
                                        probSum[1] += probVals[1]
                                    probSum[0] /= index + 1
                                    probSum[1] /= index + 1

                                    if (probSum[truthValue] > probSum[abs(truthValue-1)]):
                                        (reliability[source["domain"]])[0] += 1#correct
                                    else:
                                        (reliability[source["domain"]])[1] += 1#incorrect
                            except Exception as e:
                                # raise e
                                continue

            with open("../resources//reliability//out/" + file_, "w") as fp:
                for r in reliability:
                    articleStances = reliability[r]
                    percentCorrect = articleStances[0]/(articleStances[0]+ articleStances[1])
                    fp.write(r + "\t" + str(percentCorrect) + "\t" + str(articleStances) + "\n")
    return



def combineReliabilityScores():
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

    sortme = []
    for x in reliability:
        completeDict[x] = reliability[x][0]/(reliability[x][0]+reliability[x][1]) 
        sortme.append((x, reliability[x][0]/(reliability[x][0]+reliability[x][1]), reliability[x][0], reliability[x][1]))

    # sortme.sort(key=lambda x:x[2]+x[3])
    # for m in sortme:
    #     print(m)

    # for domain in reliability.items():
    #     if ((domain[1])[0] +(domain[1])[1]) >= 3:
    #         myList.append(domain)

    # myList.sort(key=lambda x: ((x[1])[0]*1.0)/((x[1])[0] +(x[1])[1]))


    with open("compiledReliabilityDictFINAL.txt", "w") as a:
        a.write(json.dumps(completeDict))


if __name__ == "__main__":
    # with Pool(processes=6) as pool:
    #     procs = []
    #     procs.append(pool.apply_async(determineRel,("../resources//reliability//one",)))
    #     procs.append(pool.apply_async(determineRel,("../resources//reliability//two",)))
    #     procs.append(pool.apply_async(determineRel,("../resources//reliability//three",)))
    #     procs.append(pool.apply_async(determineRel,("../resources//reliability//four",)))
    #     procs.append(pool.apply_async(determineRel,("../resources//reliability//five",)))
    #     procs.append(pool.apply_async(determineRel,("../resources//reliability//six",)))

    #     #wait for each process to finish
    #     for proc in procs:
    #         proc.wait()

    combineReliabilityScores()
