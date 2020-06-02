# Truth-Finder

## Description
Due to the increasing amount of 'fake news' being spread on the internet, it is vital that we start to develop methods
to sift through texual articles and authenticate the credibility of the information in them.
 
The purpose of this program is to evaluate the validity of a given textual claim and be able to provide an explanation as to why the information is true or false.
It accomplishes this task by querying the claim in a search engine and assessing the articles that either support or refute the claim. It then inteligently reads and evauluates the credibility of the supporting and refuting articles. To make it's final classification it aggregates the overall 'opinion' it has found across the articles weighting the opinions by it's derived opinion of source reliability. The texual explanation is the most relevent snippet of text from one of the articles which had the 'correct' opinion regarding the claim. 


The program is based off of the techniques proposed by Popat et. al [[1](#References)] in "Where the Truth Lies: Explaining the Credibility of Emerging Claims on the Web and Social Media".

## Example Queries

query: "Obama was born in Kenya"

![query1](imgs/query1.png)

## Dependancies
- Python 3.6+
- googlesearch
- BeautifulSoup (bs4)
- NLTK
  - PorterStemmer
- tqdm
- liblinear
  - https://www.csie.ntu.edu.tw/~cjlin/liblinear/

## Usage

### Using pre-trained models

I have included model weights that I have trained in the resources folder.

```
python -m truth_finder \
    --credModel resources/models/distantSupervision.model \
    --lingFeats resources/lingfeats.pkl \
    --stanceModel resources/models/stanceV1.model \
    --stanceFeats resources/stanceFeats.pkl \
    --srcRel resources/reliabilityDict.json
```

### Training new models

Traing new models is more complicated as you must first aquire the neccessary data for the models.
Please read the [Data](#Data) section prior to performing these steps.

Train the  stance determination model:

```
python -m train_stance \
    --featPath resources/stanceFeats.pkl \
    --dataPath resources/Snopes/ \
    --stancePath resources/models/stance.model
```

Contruct Training Data:
```
python -m BuildContentAwareData
```

Train the credibility classifier:
```
python -m train_credibility_model --modelName distantSupervision.model
```

## Data

The program utilizes a dataset compiled by researchers at the "Max Planck Institute for Informatics". It contains articles from the popular factchecking website 'Snopes.com' with related webseach results. 

Please download the dataset and exact the contents in the resources/ directory of the TruthFinder program.
[http://resources.mpi-inf.mpg.de/impact/web_credibility_analysis/Snopes.tar.gz](http://resources.mpi-inf.mpg.de/impact/web_credibility_analysis/Snopes.tar.gz)


## References
<a id="1">[1]</a> 
Kashyap Popat. et. al (2017).
Where the Truth Lies: Explaining the Credibility of Emerging Claims on the Web and Social Media<br>
Proceedings of the 26th International Conference on World Wide Web Companion, April 03-07 <br>
https://resources.mpi-inf.mpg.de/impact/web_credibility_analysis/www2017_popat.pdf

