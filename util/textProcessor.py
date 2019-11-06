from bs4 import BeautifulSoup
import requests
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk import ngrams
from collections import defaultdict

def pullArticleText(webAddress):
    """
        Return: list of text representing each section of webpage

        param: webAddress: the url to be accessed
    """
    articleText = []
    try:
        webSource = requests.get(webAddress).text
    except Exception as e:
        raise e

    soup = BeautifulSoup(webSource, 'lxml')
    for article in soup.find_all('div'):
        try:
            articleText.append(article.text)

            # articleText.append(article.find('p').text)
        except:
            continue
    return articleText


def calcOverlap(claim, chunk):
    """
        Calculates the percent of the chunk that overlaps with the claim

        returns float representing percent overlap
    """
    overlap = 0.0

    claimWords = [word.lower() for word in word_tokenize(claim) if word.isalpha()]
    chunkWords = [word.lower() for word in word_tokenize(chunk) if word.isalpha()]

    claimUnigrams = []
    claimBigrams = []
    chunkUnigrams = []
    chunkBigrams = []
    for gram in ngrams(claimWords,n=1):
        claimUnigrams.append(gram)
    for gram in ngrams(claimWords,n=2):
        claimBigrams.append(gram)
    for gram in ngrams(chunkWords,n=1):
        chunkUnigrams.append(gram)
    for gram in ngrams(chunkWords,n=2):
        chunkBigrams.append(gram)

    uniOverlap = 0.0
    biOverlap = 0.0

    for unigram in chunkUnigrams:
        if unigram in claimUnigrams:
            uniOverlap += 1.0
    for bigram in chunkBigrams:
        if bigram in claimBigrams:
            biOverlap += 1.0

    try:
        overlap = (uniOverlap + biOverlap) / (len(chunkUnigrams) + len(chunkBigrams))
    except:
        overlap = 0.0
        pass
    return overlap

def getSnippets(textSections, maxlen=1, claim = ""):
    """
    Return list of all possible snippets of text ranging from 1 sentence to 'maxlen' sentences

    param:textSections: list of sections of text to be processed into snippets
        Each string in list will be treated as a separate text. Will not combine
        sentences from different texts
    param: maxlen: the maximum number of sentences each snippet will contain (default 0)
    paran: claim: if wanting only snippets that are relevent to a specific chunk
                    input chunk here
    """
    snippets = []

    for section in textSections:
        sectionSnips = defaultdict(lambda : [])
        for sentence in sent_tokenize(section):
            sectionSnips[1].append(sentence)                     

        if(maxlen>1):
            for snipLength in range(2, maxlen+1):#make list of sentence of each size requested
                index = 0
                if snipLength <= len(sectionSnips[1]):
                    while index < len(sectionSnips[1]):#loop through all previously read sentences
                        sip = (sectionSnips[1])[index]
                        activeIndex = index + 1
                        while activeIndex-index < snipLength and activeIndex<len(sectionSnips[1]):
                            sip += " " + (sectionSnips[1])[activeIndex]
                            activeIndex += 1
                        if activeIndex-index == snipLength:
                            sectionSnips[snipLength].append(sip)
                        index += 1
        for snipGroup in sectionSnips.keys():
            for snip in sectionSnips[snipGroup]:
                snippets.append(snip)

    if claim != "":
        relevent = set()
        for snip in snippets:
            if calcOverlap(claim, snip) >= 0.4:
                relevent.add(snip)
        return list(relevent)

    return snippets


def prepTextForClassification(text, featDict):
    """
        Returns bigrams and unigram from text. Usable to define problem for linlinear predictor

        param: text: string to be processed 
        param: features: dictionary of possible features for the model
    """
    wordBag = word_tokenize(text)
    wordBag = [word.lower() for word in wordBag if word.isalpha()]

    #get uni/bigrams
    unigrams = ngrams(wordBag,n=1)
    bigrams = ngrams(wordBag,n=2)
    
    #add data to training/testing Sample
    features = defaultdict(lambda:0)
    for gram in unigrams:
        if featDict[gram] != 0:#don't count gram that aren't known features
            features[featDict[gram]] += 1
    for gram in bigrams:
        if featDict[gram] != 0:#don't count gram that aren't known features
            features[featDict[gram]] += 1

    return dict(features)