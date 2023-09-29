---
hide:
  - navigation
---

## **Version 0.8.1**
*Release date: 29 September, 2023*

* Remove unnecessary print statements

## **Version 0.8.0**
*Release date: 29 September, 2023*

**Highlights**:

* Use `KeyLLM` to leverage LLMs for extracting keywords
  * Use it either with or without candidate keywords generated through `KeyBERT`
  * Multiple LLMs are integrated: OpenAI, Cohere, LangChain, HF, and LiteLLM  

```python
import openai
from keybert.llm import OpenAI
from keybert import KeyLLM

# Create your LLM
openai.api_key = "sk-..."
llm = OpenAI()

# Load it in KeyLLM
kw_model = KeyLLM(llm)
```

See [here](https://maartengr.github.io/KeyBERT/guides/keyllm.html) for full documentation on use cases of `KeyLLM` and [here](https://maartengr.github.io/KeyBERT/guides/llms.html) for the implemented Large Language Models.

**Fixes**:

* Enable Guided KeyBERT for seed keywords differing among docs by [@shengbo-ma](https://github.com/shengbo-ma) in [#152](https://github.com/MaartenGr/KeyBERT/pull/152)


## **Version 0.7.0**
*Release date: 3 November, 2022*

**Highlights**:

* Cleaned up documentation and added several visual representations of the algorithm (excluding MMR / MaxSum)
* Added function to extract and pass word- and document embeddings which should make fine-tuning much faster

```python
from keybert import KeyBERT

kw_model = KeyBERT()

# Prepare embeddings
doc_embeddings, word_embeddings = kw_model.extract_embeddings(docs)

# Extract keywords without needing to re-calculate embeddings
keywords = kw_model.extract_keywords(docs, doc_embeddings=doc_embeddings, word_embeddings=word_embeddings)
```

Do note that the parameters passed to `.extract_embeddings` for creating the vectorizer should be exactly the same as those in `.extract_keywords`. 

**Fixes**:

* Redundant documentation was removed by [@mabhay3420](https://github.com/priyanshul-govil) in [#123](https://github.com/MaartenGr/KeyBERT/pull/123)
* Fixed Gensim backend not working after v4 migration ([#71](https://github.com/MaartenGr/KeyBERT/issues/71))
* Fixed `candidates` not working ([#122](https://github.com/MaartenGr/KeyBERT/issues/122))


## **Version 0.6.0**
*Release date: 25 July, 2022*

**Highlights**:

* Major speedup, up to 2x to 5x when passing multiple documents (for MMR and MaxSum) compared to single documents
* Same results whether passing a single document or multiple documents
* MMR and MaxSum now work when passing a single document or multiple documents
* Improved documentation
* Added 🤗 Hugging Face Transformers

```python
from keybert import KeyBERT
from transformers.pipelines import pipeline

hf_model = pipeline("feature-extraction", model="distilbert-base-cased")
kw_model = KeyBERT(model=hf_model)
```

* Highlighting support for Chinese texts
    * Now uses the `CountVectorizer` for creating the tokens
    * This should also improve the highlighting for most applications and higher n-grams

![image](https://user-images.githubusercontent.com/25746895/179488649-3c66403c-9620-4e12-a7a8-c2fab26b18fc.png)

**NOTE**: Although highlighting for Chinese texts is improved, since I am not familiar with the Chinese language there is a good chance it is not yet as optimized as for other languages. Any feedback with respect to this is highly appreciated!

**Fixes**: 

* Fix typo in ReadMe by [@priyanshul-govil](https://github.com/priyanshul-govil) in [#117](https://github.com/MaartenGr/KeyBERT/pull/117)
* Add missing optional dependencies (gensim, use, and spacy) by [@yusuke1997](https://github.com/yusuke1997)
 in [#114](https://github.com/MaartenGr/KeyBERT/pull/114)



## **Version 0.5.1**
*Release date:  31 March, 2022*


* Added a [page](https://maartengr.github.io/KeyBERT/guides/countvectorizer.html) about leveraging `CountVectorizer` and `KeyphraseVectorizers`
    * Shoutout to [@TimSchopf](https://github.com/TimSchopf) for creating and optimizing the package!
    * The `KeyphraseVectorizers` package can be found [here](https://github.com/TimSchopf/KeyphraseVectorizers)
* Fixed Max Sum Similarity returning incorrect similarities [#92](https://github.com/MaartenGr/KeyBERT/issues/92)
    * Thanks to [@kunihik0](https://github.com/kunihik0) for the PR!
* Fixed out of bounds condition in MMR
    * Thanks to [@artmatsak](https://github.com/artmatsak) for the PR!
* Started styling with Flake8 and Black (which was long overdue)
    * Added pre-commit to make following through a bit easier with styling

## **Version 0.5.0**
*Release date:  28 September, 2021*

**Highlights**:

* Added Guided KeyBERT
    * kw_model.extract_keywords(doc, seed_keywords=seed_keywords)
    * Thanks to [@zolekode](https://github.com/zolekode) for the inspiration!
* Use the newest all-* models from SBERT

**Miscellaneous**:

* Added instructions in the FAQ to extract keywords from Chinese documents
* Fix typo in ReadMe by [@koaning](https://github.com/koaning) in [#51](https://github.com/MaartenGr/KeyBERT/pull/51)


## **Version 0.4.0**
*Release date:  23 June, 2021*

**Highlights**:

* Highlight a document's keywords with:
    * ```keywords = kw_model.extract_keywords(doc, highlight=True)```
* Use `paraphrase-MiniLM-L6-v2` as the default embedder which gives great results!

**Miscellaneous**:

* Update Flair dependencies
* Added FAQ

## **Version 0.3.0**
*Release date:  10 May, 2021*

The two main features are **candidate keywords**
and several **backends** to use instead of Flair and SentenceTransformers!

**Highlights**:

* Use candidate words instead of extracting those from the documents ([#25](https://github.com/MaartenGr/KeyBERT/issues/25))
    * ```KeyBERT().extract_keywords(doc, candidates)```
* Spacy, Gensim, USE, and Custom Backends were added (see documentation [here](https://maartengr.github.io/KeyBERT/guides/embeddings.html))

**Fixes**:

* Improved imports
* Fix encoding error when locally installing KeyBERT ([#30](https://github.com/MaartenGr/KeyBERT/issues/30))

**Miscellaneous**:

* Improved documentation (ReadMe & MKDocs)
* Add the main tutorial as a shield
* Typos ([#31](https://github.com/MaartenGr/KeyBERT/pull/31), [#35](https://github.com/MaartenGr/KeyBERT/pull/35))


## **Version 0.2.0**
*Release date:  9 Feb, 2021*

**Highlights**:

* Add similarity scores to the output
* Add Flair as a possible back-end
* Update documentation + improved testing

## **Version 0.1.2**
*Release date:  28 Oct, 2020*

Added Max Sum Similarity as an option to diversify your results.


## **Version 0.1.0**
*Release date:  27 Oct, 2020*

This first release includes keyword/keyphrase extraction using BERT and simple cosine similarity.
There is also an option to use Maximal Marginal Relevance to select the candidate keywords/keyphrases.
