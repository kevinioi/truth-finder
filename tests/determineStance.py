import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

# from nltk.tokenize import word_tokenize
# from liblinearpkg import *
# from liblinearpkg import liblinearutil as llu
# from collections import defaultdict
# from util import featureBag
# from util import textProcessor

# text = textProcessor.pullArticleText("https://www.truthorfiction.com/41-lb-rat-captured-in-new-york-city/")
# claim = "A photograph shows a 41 lb rat that was caught in New York City."
# textSnips = textProcessor.getSnippets(text, 4)
# text = None

# relevent = set()

# for snippet in textSnips:
#     if textProcessor.calcOverlap(claim, snippet) >= 0.4:
#         relevent.add(snippet)

# # featDict = featureBag.getFeatureFile("../resources/feats.pickle")
# # model1 = llu.load_model("../resources//stance_models/model.liblin")

# for r in relevent:
#     print("***************")
#     print(r)
#     break
#     # p_labels, p_acc, p_vals = llu.predict( [0], [textProcessor.prepTextForClassification(r,featDict)],model1, '-b 1')

from util import featureBag
featureBag.createFeatureFile('../resources//snopesData/', '../resources//featsV2.pickle')