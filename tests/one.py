import sys
sys.path.append('../')
from util import featureBag
from collections import defaultdict

# myDict = featureBag.getFeatureFile("../resources/feats.pickle")

# for i, x in enumerate(myDict):
#     print(f'{x}: {myDict[x]}')
#     if i > 5:
#         break
# print(myDict[("rumor",)])
# print(len(myDict)+7)

# for x in myDict:
#     if myDict[x] == 154762:
#         print(f'{x}: 154762, refute')
#     if myDict[x] == 154763:
#         print(f'{x}: 154763, refute')
#     if myDict[x] == 1739:
#         print(f'{x}: 1739, support')

# featureBag.createFeatureFile('../../data/', '../resources/feats.pickle')


myDict = {"k":1,"d":2,"r":3}

for x in myDict.values():
    print(x)

