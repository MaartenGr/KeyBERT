---
hide:
  - navigation
---

## **Which embedding model works best for which language?**
Unfortunately, there is not a definitive list of the best models for each language, this highly depends
on your data, the model, and your specific use-case. However, the default model in KeyBERT
(`"all-MiniLM-L6-v2"`) works great for **English** documents. In contrast, for **multi-lingual**
documents or any other language, `"paraphrase-multilingual-MiniLM-L12-v2""` has shown great performance.

If you want to use a model that provides a higher quality, but takes more compute time, then I would advise using `paraphrase-mpnet-base-v2` and `paraphrase-multilingual-mpnet-base-v2` instead.


## **Should I preprocess the data?**
No. By using document embeddings there is typically no need to preprocess the data as all parts of a document
are important in understanding the general topic of the document. Although this holds true in 99% of cases, if you
have data that contains a lot of noise, for example, HTML-tags, then it would be best to remove them. HTML-tags
typically do not contribute to the meaning of a document and should therefore be removed. However, if you apply
topic modeling to HTML-code to extract topics of code, then it becomes important.


## **How can I speed up the model?**
Since KeyBERT uses large language models as its backend, a GPU is typically prefered when using this package. 
Although it is possible to use it without a dedicated GPU, the inference speed will be significantly slower.

A second method for speeding up KeyBERT is by passing it multiple documents at once. By doing this, words 
need to only be embedded a single time, which can result in a major speed up. 

This is **faster**:

```python
from keybert import KeyBERT

kw_model = KeyBERT()

keywords = kw_model.extract_keywords(my_list_of_documents)
```

This is **slower**:

```python
from keybert import KeyBERT

kw_model = KeyBERT()

keywords = []
for document in my_list_of_documents:
    keyword = kw_model.extract_keywords(document)
    keywords.append(keyword)
```


## **How can I use KeyBERT with Chinese documents?**
You need to make sure you use a tokenizer in KeyBERT that supports tokenization of Chinese. I suggest installing [`jieba`](https://github.com/fxsjy/jieba) for this:

```python
from sklearn.feature_extraction.text import CountVectorizer
import jieba

def tokenize_zh(text):
    words = jieba.lcut(text)
    return words

vectorizer = CountVectorizer(tokenizer=tokenize_zh)
```

Then, simply pass the vectorizer to your KeyBERT instance:

```python
from keybert import KeyBERT

kw_model = KeyBERT()
keywords = kw_model.extract_keywords(doc, vectorizer=vectorizer)
```

It also supports highlighting:

![image](https://user-images.githubusercontent.com/25746895/179488649-3c66403c-9620-4e12-a7a8-c2fab26b18fc.png)
