import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

import json

tCount = 0
fCount = 0

#load data from all source files
for file_ in os.listdir("../resources//contentTrain//used_claims"):
    if file_.endswith(".json"):
        with open("../resources//contentTrain//used_claims/" + file_, 'r') as doc:#read snopes file
            fileData =  json.loads(doc.read())

        if fileData['Credibility'] == 'false' or fileData['Credibility'] == 'mostly false':
            fCount += 1
        else:
            tCount += 1

print(f"True: {tCount}")
print(f"False: {fCount}")
