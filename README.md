[![PyPI - Python](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue.svg)](https://pypi.org/project/keybert/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/MaartenGr/keybert/blob/master/LICENSE)
[![PyPI - PyPi](https://img.shields.io/pypi/v/keyBERT)](https://pypi.org/project/keybert/)
[![Build](https://img.shields.io/github/workflow/status/MaartenGr/keyBERT/Code%20Checks/master)](https://pypi.org/project/keybert/)

<img src="images/logo.png" width="35%" height="35%" align="right" />

# KeyBERT

KeyBERT is a minimal and easy-to-use keyword extraction technique that leverages BERT embeddings to
create keywords and keyphrases that are most similar to a document. 

Corresponding medium post can be found [here]().

<a name="toc"/></a>
## Table of Contents  
<!--ts-->
   1. [About the Project](#about)  
   2. [Getting Started](#gettingstarted)    
        2.1. [Installation](#installation)    
        2.2. [Basic Usage](#usage)   
        2.3. [Overview](#overview)    
<!--te-->


<a name="about"/></a>
## 1. About the Project
[Back to ToC](#toc)  

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
    
**NOTE I**: If you find a paper or github repo that has an easy-to-use implementation
of BERT-embeddings for keyword/keyphrase extraction, let me know! I'll make sure to
add it a reference to this repo. 

**NOTE II**: If you use MMR to select the candidates instead of simple consine similarity,
this repo is essentially a simplified implementation of 
[EmbedRank](https://github.com/swisscom/ai-research-keyphrase-extraction) 
with BERT-embeddings. 

## References
Below, you can find several resources that were used for the creation of KeyBERT 
but most importantly, are amazing resources for creating impressive keyword extraction models: 

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

**NOTE**: If you find a paper or github repo that is interesting 