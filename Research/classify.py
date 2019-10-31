#https://www.youtube.com/watch?v=zi16nl82AMA&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=11

import nltk
import random


train_sample = []
test_sample = []




# using naive bayes
# posterior = (prior*likelyhood)/ evidence

classifier = nltk.NaiveBayesClassifier.train(train_sample)
classifier.show_most_informative_features(15)