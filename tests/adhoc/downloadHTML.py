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

countSnopes = 0
articleCount = 0
referenceFile =  open('htmlref.txt', "w")

#load data from all source files
for file_ in os.listdir("../resources//partialSnopes"):    
    truthValue = None
    print("********************************************************")
    print(file_)
    print(countSnopes)
    countSnopes+=1

    if file_.endswith(".json"):
        with open("../resources//partialSnopes/" + file_, 'r') as doc:
            fileData =  json.loads(doc.read())

        if fileData['Credibility'] == 'false' or fileData['Credibility'] == 'mostly false':
            truthValue = 0
        else:
            truthValue = 1

        count = 0

        for page in fileData["Google Results"]:#load page of google results
            for resultsDict in page.values():#load sources from google page
                for source in resultsDict:#process each source
                    if (source["domain"] != "www.snopes.com"):
                        print(source["domain"])
                        try:
                            articleCount +=1
                            text = textProcessor.pullArticleText(source["link"],timeoutTime=6)
                            referenceFile.write(source["link"] + "\t"+str(articleCount))
                            with open("../resources//webarticles/" + str(articleCount)+".txt", "w") as articleFile:
                                for section in text:
                                    try:
                                        articleFile.write(section)
                                        articleFile.write('\n')
                                    except:
                                        continue

                        except Exception as e:
                            raise e
                            # continue
            #         break#each entry in page
            #     break#each page?
            # break #each page.
        # break#each file
    # break#each file?
