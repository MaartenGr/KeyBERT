[![PyPI - Python](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue.svg)](https://pypi.org/project/keybert/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MaartenGr/keybert/blob/master/LICENSE)
[![PyPI - PyPi](https://img.shields.io/pypi/v/keyBERT)](https://pypi.org/project/keybert/)
[![Build](https://img.shields.io/github/workflow/status/MaartenGr/keyBERT/Code%20Checks/master)](https://pypi.org/project/keybert/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OxpgwKqSzODtO3vS7Xe1nEmZMCAIMckX?usp=sharing)

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
        2.5. [Embedding Models](#embeddings)
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
Installation can be done using [pypi](https://pypi.org/project/keybert/):

```
pip install keybert
```

You may want to install more depending on the transformers and language backends that you will be using. The possible installations are:

```
pip install keybert[flair]
pip install keybert[gensim]
pip install keybert[spacy]
pip install keybert[use]
```

To install all backends:

```
pip install keybert[all]
```

<a name="usage"/></a>
###  2.2. Usage

The most minimal example can be seen below for the extraction of keywords:
```python
from keybert import KeyBERT

doc = """
         Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal). 
         A supervised learning algorithm analyzes the training data and produces an inferred function, 
         which can be used for mapping new examples. An optimal scenario will allow for the 
         algorithm to correctly determine the class labels for unseen instances. This requires 
         the learning algorithm to generalize from the training data to unseen situations in a 
         'reasonable' way (see inductive bias).
      """
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(doc)
```

You can set `keyphrase_ngram_range` to set the length of the resulting keywords/keyphrases:

```python
>>> kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words=None)
[('learning', 0.4604),
 ('algorithm', 0.4556),
 ('training', 0.4487),
 ('class', 0.4086),
 ('mapping', 0.3700)]
```

To extract keyphrases, simply set `keyphrase_ngram_range` to (1, 2) or higher depending on the number 
of words you would like in the resulting keyphrases: 

```python
>>> kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words=None)
[('learning algorithm', 0.6978),
 ('machine learning', 0.6305),
 ('supervised learning', 0.5985),
 ('algorithm analyzes', 0.5860),
 ('learning function', 0.5850)]
``` 

We can highlight the keywords in the document by simply setting `hightlight`:

```python
keywords = kw_model.extract_keywords(doc, highlight=True)
```
<img src="images/highlight.png" width="75%" height="75%" />
  
  
**NOTE**: For a full overview of all possible transformer models see [sentence-transformer](https://www.sbert.net/docs/pretrained_models.html).
I would advise either `"paraphrase-MiniLM-L6-v2"` for English documents or `"paraphrase-multilingual-MiniLM-L12-v2"` 
for multi-lingual documents or any other language.  

<a name="maxsum"/></a>
###  2.3. Max Sum Similarity

To diversify the results, we take the 2 x top_n most similar words/phrases to the document.
Then, we take all top_n combinations from the 2 x top_n words and extract the combination 
that are the least similar to each other by cosine similarity.

```python
>>> kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english', 
                              use_maxsum=True, nr_candidates=20, top_n=5)
[('set training examples', 0.7504),
 ('generalize training data', 0.7727),
 ('requires learning algorithm', 0.5050),
 ('supervised learning algorithm', 0.3779),
 ('learning machine learning', 0.2891)]
``` 


<a name="maximal"/></a>
###  2.4. Maximal Marginal Relevance

To diversify the results, we can use Maximal Margin Relevance (MMR) to create
keywords / keyphrases which is also based on cosine similarity. The results 
with **high diversity**:

```python
>>> kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english', 
                              use_mmr=True, diversity=0.7)
[('algorithm generalize training', 0.7727),
 ('labels unseen instances', 0.1649),
 ('new examples optimal', 0.4185),
 ('determine class labels', 0.4774),
 ('supervised learning algorithm', 0.7502)]
``` 

The results with **low diversity**:  

```python
>>> kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english', 
                              use_mmr=True, diversity=0.2)
[('algorithm generalize training', 0.7727),
 ('supervised learning algorithm', 0.7502),
 ('learning machine learning', 0.7577),
 ('learning algorithm analyzes', 0.7587),
 ('learning algorithm generalize', 0.7514)]
``` 


<a name="embeddings"/></a>
###  2.5. Embedding Models
KeyBERT supports many embedding models that can be used to embed the documents and words:

* Sentence-Transformers
* Flair
* Spacy
* Gensim
* USE

Click [here](https://maartengr.github.io/KeyBERT/guides/embeddings.html) for a full overview of all supported embedding models.

**Sentence-Transformers**  
You can select any model from `sentence-transformers` [here](https://www.sbert.net/docs/pretrained_models.html) 
and pass it through KeyBERT with `model`:

```python
from keybert import KeyBERT
kw_model = KeyBERT(model='paraphrase-MiniLM-L6-v2')
```

Or select a SentenceTransformer model with your own parameters:

```python
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
kw_model = KeyBERT(model=sentence_model)
```

**Flair**  
[Flair](https://github.com/flairNLP/flair) allows you to choose almost any embedding model that 
is publicly available. Flair can be used as follows:

```python
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings

roberta = TransformerDocumentEmbeddings('roberta-base')
kw_model = KeyBERT(model=roberta)
```

You can select any 🤗 transformers model [here](https://huggingface.co/models).


## Citation
To cite PolyFuzz in your work, please use the following bibtex reference:

```bibtex
@misc{grootendorst2020keybert,
  author       = {Maarten Grootendorst},
  title        = {KeyBERT: Minimal keyword extraction with BERT.},
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.3.0},
  doi          = {10.5281/zenodo.4461265},
  url          = {https://doi.org/10.5281/zenodo.4461265}
}
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
The selection of keywords/keyphrases was modeled after:
* https://github.com/swisscom/ai-research-keyphrase-extraction

**NOTE**: If you find a paper or github repo that has an easy-to-use implementation
of BERT-embeddings for keyword/keyphrase extraction, let me know! I'll make sure to
add a reference to this repo. 

