# CountVectorizer Tips & Tricks

An unexpectly important component of KeyBERT is the CountVectorizer. In KeyBERT, it is used to split up your documents into candidate keywords and keyphrases. However, 
there is much more flexibility with the CountVectorizer than you might have initially thought. Since we use the vectorizer to split up the documents *after* embedding them, 
we can parse the document however we want as it does not affect the quality of the document embeddings. In this page, we will go through several examples of how you can take 
the CountVectorizer to the next level and improve upon the generated keywords. 


## **Basic Usage**

First, let's start with defining our text and the keyword model:

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
kw_model = KeyBERT()
```

We will use the above code throughout this tutorial as the base and built upon it with the CountVectorizer. 
Next, we can use a basic vectorizer when extracting keywords as follows:


```python
>>> vectorizer = CountVectorizer()
>>> keywords = kw_model.extract_keywords(doc, vectorizer=vectorizer)
[('learning', 0.4604),
 ('algorithm', 0.4556),
 ('training', 0.4487),
 ('class', 0.4086),
 ('mapping', 0.3700)]
```

**NOTE**: Although I typically like to use `use_mmr=True` as it often improves upon the generated keywords, this tutorial will do without 
in order give you a clear view of the effects of the CountVectorizer. 

## **Parameters**

There are a number of basic parameters in the CountVectorizer that we can use to improve upon the quality of the resulting keywords. 

### ngram_range

By setting the `ngram_range` we can decide how many tokens the keyphrases needs to be as a minimum and how long it can be as a maximum:

```python
>>> vectorizer = CountVectorizer(ngram_range=(1, 3))
>>> kw_model.extract_keywords(doc, vectorizer=vectorizer)
[('supervised learning is', 0.7048),
 ('supervised learning algorithm', 0.6834),
 ('supervised learning', 0.6658),
 ('supervised', 0.6523),
 ('in supervised learning', 0.6474)]
```

As we can see, the length of the resulting keyphrases are higher than what we have seen before. This may happen as embeddings in vector space are often closer 
in distance if their document counterparts in similar in size. 

There are two interesting things happening here. First, there are many similar keyphrases that we want to diversify, which we can achieve by setting `use_mmr=True`. Second, 
you may have noticed stopwords appearing in the keyphrases. That we can solve by following the section below!

### stop_words

As we have seen in the results above, stopwords might appear in your keyphrases. To remove them, we can tell the CountVectorizer to either remove a list of keywords 
that we supplied ourselves or simply state for which language stopwords need to be removed:

```python
>>> vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words="english")
>>> kw_model.extract_keywords(doc, vectorizer=vectorizer)
[('supervised learning algorithm', 0.6834),
 ('supervised learning', 0.6658),
 ('supervised learning example', 0.6641),
 ('supervised learning machine', 0.6528),
 ('function labeled training', 0.6526)]
```

This already looks much better! The stopwords are removed and the resulting keyphrases already look a bit more interesting and useful.

### vocabulary

For some use cases, keywords can only be generated from predefined vocabularies. For example, when you already have a list of possible keywords you can use those as a vocabulary and 
ask the CountVectorizer to only select keywords from that list. 

First, let's define our vocabulary:

```python
vocab = [
 'produces inferred function',
 'supervised', 
 'inductive', 
 'function', 
 'bias', 
 'supervisory', 
 'supervised learning',
 'infers function',
 'supervisory signal',
 'inductive bias',
 'unseen instances']
```

Then, we pass that vocabulary to our CountVectorizer and extract our keywords:

```python
>>> vectorizer = CountVectorizer(ngram_range=(1, 3), vocabulary=vocab)
>>> kw_model.extract_keywords(doc, vectorizer=vectorizer)
[('supervised learning', 0.6658),
 ('supervised', 0.6523),
 ('supervisory signal', 0.357),
 ('inductive bias', 0.3377),
 ('produces inferred function', 0.3365)]
```

## **KeyphraseVectorizers**

To even further enhance the possibilities of the CountVectorizer, [Tim Schopf](https://github.com/TimSchopf) created an excellent package, [KeyphraseVectorizers](https://github.com/TimSchopf/KeyphraseVectorizers), that enriches the CountVectorizer with the possibilities to extract keyphrases with part-of-speech patterns using the Spacy library. 

The great thing about the `KeyphraseVectorizers` is that aside from leveraging part-of-speech patterns, it automatically extract keyphrases without the need to specify 
an n-gram range. That by itself is an amazing feature to have! Other advantages of this package:

* Extract grammatically accurate keyphases based on their part-of-speech tags.
* No need to specify n-gram ranges.
* Get document-keyphrase matrices.
* Multiple language support.
* User-defined part-of-speech patterns for keyphrase extraction possible.


### Usage

First, we need to install the package:

```bash
pip install keyphrase-vectorizers
```

Then, let's see what the output looks like with the basic `CountVectorizer` using a larger n-gram value:

```python
>>> vectorizer = CountVectorizer(ngram_range=(1, 3))
>>> kw_model.extract_keywords(doc, vectorizer=vectorizer)
[('supervised learning is', 0.7048),
 ('supervised learning algorithm', 0.6834),
 ('supervised learning', 0.6658),
 ('supervised', 0.6523),
 ('in supervised learning', 0.6474)]
```

Not bad but as we have seen before, this can definitely be improved. Now, let's use the `KeyphraseCountVectorizer` instead:

```python
>>> from keyphrase_vectorizers import KeyphraseCountVectorizer
>>> vectorizer = KeyphraseCountVectorizer()
>>> kw_model.extract_keywords(doc, vectorizer=vectorizer)
[('supervised learning algorithm', 0.6834),
 ('supervised learning', 0.6658),
 ('learning algorithm', 0.5549),
 ('training data', 0.511),
 ('training', 0.3858)]
```

A large improvement compared to the basic CountVectorizer! Now, as seen before it seems that there are still some repeated keyphrases that we want to remove. To do this, 
we again leverage the `MMR` function on top of KeyBERT to diversify the output:

```python
>>> from keyphrase_vectorizers import KeyphraseCountVectorizer
>>> vectorizer = KeyphraseCountVectorizer()
>>> kw_model.extract_keywords(doc, vectorizer=vectorizer, use_mmr=True)
[('supervised learning algorithm', 0.6834),
 ('unseen instances', 0.3246),
 ('supervisory signal', 0.357),
 ('inductive bias', 0.3377),
 ('class labels', 0.3715)]
```

We can see much more diverse keyphrases and based on the input document the keyphrases also make sense. 


### Languages

Those familiar with Spacy might know that in order to use part-of-speech, we need a language-specific model. 
You can find an overview of these models [here](https://spacy.io/models). To change the language model, we only need to change 
one parameter in order to select a different language:

```python
vectorizer = KeyphraseCountVectorizer(spacy_pipeline='de_core_news_sm')
```


### Part-of-speech

KeyphraseVectorizers extracts the part-of-speech tags from the documents and then applies a regex pattern to extract 
keyphrases that fit within that pattern. The default pattern is `<J.*>*<N.*>+` which means that it extract keyphrases 
that have 0 or more adjectives followed by 1 or more nouns. 

However, we might not agree with that for our specific use case! Fortunately, the package allows you to use a different 
pattern. To visualize the effect, let's first perform it with the default settings:

```python
>>> vectorizer = KeyphraseCountVectorizer()
>>> kw_model.extract_keywords(doc, vectorizer=vectorizer)
[('supervised learning algorithm', 0.6834),
 ('supervised learning', 0.6658),
 ('learning algorithm', 0.5549),
 ('training data', 0.511),
 ('training', 0.3858)]
```

Although the above keyphrases seem accurate, we might want to only extract a noun from the documents in 
order to only extract keywords and not keyphrases:

```python
>>> vectorizer = KeyphraseCountVectorizer(pos_pattern='<N.*>')
>>> kw_model.extract_keywords(doc, vectorizer=vectorizer)
[('learning', 0.467),
 ('training', 0.3858),
 ('labels', 0.3728),
 ('data', 0.2993),
 ('algorithm', 0.2827)]
```

These seem much better as keywords now that we focus only on nouns in the document. 