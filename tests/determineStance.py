import sys
sys.path.append('../')

from nltk.tokenize import word_tokenize
from liblinearpkg import *
from liblinearpkg import liblinearutil as llu
from util import featureBag
from util import textProcessor

text = textProcessor.pullArticleText("https://www.truthorfiction.com/41-lb-rat-captured-in-new-york-city/")
claim = "A photograph shows a 41 lb rat that was caught in New York City."
textSnips = textProcessor.getSnippets(text, 4)
text = None

relevent = set()

for snippet in textSnips:
    if textProcessor.calcOverlap(claim, snippet) >= 0.4:
        relevent.add(snippet)

textSnips = None

featDict = featureBag.getFeatureFile("../resources/featsV2.pickle")
model1 = llu.load_model("../resources/models/gen2v1.model")

probSum = [0,0]

for r in relevent:
    print("***************")
    print(r)
    p_labels, p_acc, p_vals = llu.predict( [], [textProcessor.prepTextForClassification(r,featDict)],model1, '-b 1')
    print(f"{p_labels}   {p_acc}    {p_vals}")
    probSum[0] += (p_vals[0])[0]
    probSum[1] += (p_vals[0])[1]


print(probSum)