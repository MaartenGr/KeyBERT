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
