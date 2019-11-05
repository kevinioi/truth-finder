import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')
from util import textProcessor

# text = ["Kevin is working on his assignment today. He will hopefully finish this module by the end of wednesday. It is very importanct to him that this project goes well. Alana is a very pretty girl."]

# # for x in textProcessor.getSnippets(text,4):
# #     print("*************")
# #     print(x)

# t1 = "Kevin is here"
# t2 = 'is here'

# x = textProcessor.calcOverlap(t2, t1)
# print(x)




try:
    text = textProcessor.pullArticleText("https://realpython.com/python-exceptions/#raising-an-exception")
except Exception as p:
    raise p
    #continue

