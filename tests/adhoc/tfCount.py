import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

import json

tCount = 0
fCount = 0

#load data from all source files
for file_ in os.listdir("../resources//contentTrain//filesUsed"):
    if file_.endswith(".json"):
        with open("../resources//contentTrain//filesUsed/" + file_, 'r') as doc:#read snopes file
            fileData =  json.loads(doc.read())

        if fileData['Credibility'] == 'false' or fileData['Credibility'] == 'mostly false':
            fCount += 1
        elif fileData['Credibility'] == 'true' or fileData['Credibility'] == 'mostly true':
            tCount += 1

print(f"True: {tCount}")
print(f"False: {fCount}")
