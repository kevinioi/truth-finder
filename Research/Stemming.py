#Reducing words to their root


from nltk.tokenize import word_tokenize

#there are lots of stemmers, can find another
from nltk.stem import PorterStemmer

ps = PorterStemmer()

sentence = "Kevin is a very studious student and will be going to"
words = word_tokenize(sentence)

for w in words:
    print(ps.stem(w))

