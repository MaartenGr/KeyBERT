[![PyPI - Python](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue.svg)](https://pypi.org/project/keybert/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MaartenGr/keybert/blob/master/LICENSE)
[![PyPI - PyPi](https://img.shields.io/pypi/v/keyBERT)](https://pypi.org/project/keybert/)
[![Build](https://img.shields.io/github/workflow/status/MaartenGr/keyBERT/Code%20Checks/master)](https://pypi.org/project/keybert/)

<img src="images/logo.png" width="35%" height="35%" align="right" />

# KeyBERT

KeyBERT is a minimal and easy-to-use keyword extraction technique that leverages BERT embeddings to
create keywords and keyphrases that are most similar to a document. 

Corresponding medium post can be found [here](https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea).

<a name="toc"/></a>
## Table of Contents  
<!--ts-->
   1. [About the Project](#about)  
   2. [Getting Started](#gettingstarted)    
        2.1. [Installation](#installation)    
        2.2. [Basic Usage](#usage)     
        2.3. [Max Sum Similarity](#maxsum)  
        2.4. [Maximal Marginal Relevance](#maximal)
<!--te-->


<a name="about"/></a>
## 1. About the Project
[Back to ToC](#toc)  

Although there are already many methods available for keyword generation 
(e.g., 
[Rake](https://github.com/aneesha/RAKE), 
[YAKE!](https://github.com/LIAAD/yake), TF-IDF, etc.) 
I wanted to create a very basic, but powerful method for extracting keywords and keyphrases. 
This is where **KeyBERT** comes in! Which uses BERT-embeddings and simple cosine similarity
to find the sub-phrases in a document that are the most similar to the document itself.

First, document embeddings are extracted with BERT to get a document-level representation. 
Then, word embeddings are extracted for N-gram words/phrases. Finally, we use cosine similarity 
to find the words/phrases that are the most similar to the document. The most similar words could 
then be identified as the words that best describe the entire document.  

KeyBERT is by no means unique and is created as a quick and easy method
for creating keywords and keyphrases. Although there are many great 
papers and solutions out there that use BERT-embeddings 
(e.g., 
[1](https://github.com/pranav-ust/BERT-keyphrase-extraction),
[2](https://github.com/ibatra/BERT-Keyword-Extractor),
[3](https://www.preprints.org/manuscript/201908.0073/download/final_file),
), I could not find a BERT-based solution that did not have to be trained from scratch and
could be used for beginners (**correct me if I'm wrong!**).
Thus, the goal was a `pip install keybert` and at most 3 lines of code in usage.   

<a name="gettingstarted"/></a>
## 2. Getting Started
[Back to ToC](#toc)  

<a name="installation"/></a>
###  2.1. Installation
**[PyTorch 1.2.0](https://pytorch.org/get-started/locally/)** or higher is recommended. If the install below gives an
error, please install pytorch first [here](https://pytorch.org/get-started/locally/). 

Installation can be done using [pypi](https://pypi.org/project/bertopic/):

```
pip install keybert
```

<a name="usage"/></a>
###  2.2. Usage

The most minimal example can be seen below for the extraction of keywords:
```python
from keybert import KeyBERT

doc = """
         Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs.[1] It infers a
         function from labeled training data consisting of a set of training examples.[2]
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal). 
         A supervised learning algorithm analyzes the training data and produces an inferred function, 
         which can be used for mapping new examples. An optimal scenario will allow for the 
         algorithm to correctly determine the class labels for unseen instances. This requires 
         the learning algorithm to generalize from the training data to unseen situations in a 
         'reasonable' way (see inductive bias).
      """
model = KeyBERT('distilbert-base-nli-mean-tokens')
keywords = model.extract_keywords(doc)
```

You can set `keyphrase_length` to set the length of the resulting keywords/keyphrases:

```python
>>> model.extract_keywords(doc, keyphrase_length=1, stop_words=None)
['learning', 
 'training', 
 'algorithm', 
 'class', 
 'mapping']
```

To extract keyphrases, simply set `keyphrase_length` to 2 or higher depending on the number 
of words you would like in the resulting keyphrases: 

```python
>>> model.extract_keywords(doc, keyphrase_length=2, stop_words=None)
['learning algorithm',
 'learning machine',
 'machine learning',
 'supervised learning',
 'learning function']
``` 


**NOTE**: For a full overview of all possible transformer models see [sentence-transformer](https://www.sbert.net/docs/pretrained_models.html).
I would advise either `'distilbert-base-nli-mean-tokens'` or `'xlm-r-distilroberta-base-paraphrase-v1'` as they
have shown great performance in semantic similarity and paraphrase identification respectively. 

<a name="maxsum"/></a>
###  2.3. Max Sum Similarity

To diversity the results, we take the 2 x top_n most similar words/phrases to the document.
Then, we take all top_n combinations from the 2 x top_n words and extract the combination 
that are the least similar to each other by cosine similarity.

```python
>>> model.extract_keywords(doc, keyphrase_length=3, stop_words='english', 
                           use_maxsum=True, nr_candidates=20, top_n=5)
['set training examples',
 'generalize training data',
 'requires learning algorithm',
 'superivsed learning algorithm',
 'learning machine learning']
``` 


<a name="maximal"/></a>
###  2.4. Maximal Marginal Relevance

To diversify the results, we can use Maximal Margin Relevance (MMR) to create
keywords / keyphrases which is also based on cosine similarity. The results 
with **high diversity**:

```python
>>> model.extract_keywords(doc, keyphrase_length=3, stop_words='english', use_mmr=True, diversity=0.7)
['algorithm generalize training',
 'labels unseen instances',
 'new examples optimal',
 'determine class labels',
 'supervised learning algorithm']
``` 

The results with **low diversity**:  

```python
>>> model.extract_keywords(doc, keyphrase_length=3, stop_words='english', use_mmr=True, diversity=0.2)
['algorithm generalize training',
 'learning machine learning',
 'learning algorithm analyzes',
 'supervised learning algorithm',
 'algorithm analyzes training']
``` 

## References
Below, you can find several resources that were used for the creation of KeyBERT 
but most importantly, these are amazing resources for creating impressive keyword extraction models: 

**Papers**:  
* Sharma, P., & Li, Y. (2019). [Self-Supervised Contextual Keyword and Keyphrase Retrieval with Self-Labelling.](https://www.preprints.org/manuscript/201908.0073/download/final_file)

**Github Repos**:  
* https://github.com/thunlp/BERT-KPE
* https://github.com/ibatra/BERT-Keyword-Extractor
* https://github.com/pranav-ust/BERT-keyphrase-extraction
* https://github.com/swisscom/ai-research-keyphrase-extraction

**MMR**:  
The selection of keywords/keyphrases was modelled after:
* https://github.com/swisscom/ai-research-keyphrase-extraction

**NOTE**: If you find a paper or github repo that has an easy-to-use implementation
of BERT-embeddings for keyword/keyphrase extraction, let me know! I'll make sure to
add it a reference to this repo. 

