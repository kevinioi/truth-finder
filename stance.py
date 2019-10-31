
Scikit has text feature extractor ready to use that includes TF-IDF


#training


#load all training docs
docs = []
for doc in trainingSample:
     docs.append(list(doc.words), True/False)

#shuffle the docs
random.shuffle(docs)

