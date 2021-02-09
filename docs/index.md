<img src="https://raw.githubusercontent.com/MaartenGr/KeyBERT/master/images/logo.png" width="35%" height="35%" align="right" />

# KeyBERT

KeyBERT is a minimal and easy-to-use keyword extraction technique that leverages BERT embeddings to
create keywords and keyphrases that are most similar to a document. 

## About the Project
  
Although that are already many methods available for keyword generation 
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
    
**NOTE**: If you use MMR to select the candidates instead of simple cosine similarity,
this repo is essentially a simplified implementation of 
[EmbedRank](https://github.com/swisscom/ai-research-keyphrase-extraction) 
with BERT-embeddings. 

## Installation
Installation can be done using [pypi](https://pypi.org/project/keybert/):

```
pip install keybert
```

To use Flair embeddings, install KeyBERT as follows:

```
pip install keybert[flair]
```

## Usage

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
```

You can set `keyphrase_length` to set the length of the resulting keyphras:

```python
>>> model.extract_keywords(doc, keyphrase_ngram_range=(1, 1))
[('learning', 0.4604),
 ('algorithm', 0.4556),
 ('training', 0.4487),
 ('class', 0.4086),
 ('mapping', 0.3700)]
```

To extract keyphrases, simply set `keyphrase_ngram_range` to (1, 2) or higher depending on the number 
of words you would like in the resulting keyphrases: 

```python
>>> model.extract_keywords(doc, keyphrase_ngram_range=(1, 2))
[('learning algorithm', 0.6978),
 ('machine learning', 0.6305),
 ('supervised learning', 0.5985),
 ('algorithm analyzes', 0.5860),
 ('learning function', 0.5850)]
``` 