import nltk
from nltk.tokenize import word_tokenize

sent = "Kevin is the best student in the world"
words = word_tokenize(sent)

print(nltk.pos_tag(words))