# Embedding Models
In this tutorial we will be going through the embedding models that can be used in KeyBERT.
Having the option to choose embedding models allow you to leverage pre-trained embeddings that suit your use-case.

### **Sentence Transformers**
You can select any model from sentence-transformers [here](https://www.sbert.net/docs/pretrained_models.html)
and pass it through KeyBERT with `model`:

```python
from keybert import KeyBERT
kw_model = KeyBERT(model="all-MiniLM-L6-v2")
```

Or select a SentenceTransformer model with your own parameters:

```python
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=sentence_model)
```

### **Flair**
[Flair](https://github.com/flairNLP/flair) allows you to choose almost any embedding model that
is publicly available. Flair can be used as follows:

```python
from flair.embeddings import TransformerDocumentEmbeddings

roberta = TransformerDocumentEmbeddings('roberta-base')
kw_model = KeyBERT(model=roberta)
```

You can select any 🤗 transformers model [here](https://huggingface.co/models).

Moreover, you can also use Flair to use word embeddings and pool them to create document embeddings.
Under the hood, Flair simply averages all word embeddings in a document. Then, we can easily
pass it to KeyBERT in order to use those word embeddings as document embeddings:

```python
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings

glove_embedding = WordEmbeddings('crawl')
document_glove_embeddings = DocumentPoolEmbeddings([glove_embedding])

kw_model = KeyBERT(model=document_glove_embeddings)
```

### **Spacy**
[Spacy](https://github.com/explosion/spaCy) is an amazing framework for processing text. There are
many models available across many languages for modeling text.

 allows you to choose almost any embedding model that
is publicly available. Flair can be used as follows:

To use Spacy's non-transformer models in KeyBERT:

```python
import spacy

nlp = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

kw_model = KeyBERT(model=nlp)
```

Using spacy-transformer models:

```python
import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

kw_model = KeyBERT(model=nlp)
```

If you run into memory issues with spacy-transformer models, try:

```python
import spacy
from thinc.api import set_gpu_allocator, require_gpu

nlp = spacy.load("en_core_web_trf", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
set_gpu_allocator("pytorch")
require_gpu(0)

kw_model = KeyBERT(model=nlp)
```

### **Universal Sentence Encoder (USE)**
The Universal Sentence Encoder encodes text into high dimensional vectors that are used here
for embedding the documents. The model is trained and optimized for greater-than-word length text,
such as sentences, phrases or short paragraphs.

Using USE in KeyBERT is rather straightforward:

```python
import tensorflow_hub
embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
kw_model = KeyBERT(model=embedding_model)
```

### **Gensim**
For Gensim, KeyBERT supports its `gensim.downloader` module. Here, we can download any model word embedding model
to be used in KeyBERT. Note that Gensim is primarily used for Word Embedding models. This works typically
best for short documents since the word embeddings are pooled.

```python
import gensim.downloader as api
ft = api.load('fasttext-wiki-news-subwords-300')
kw_model = KeyBERT(model=ft)
```

### **Custom Backend**
If your backend or model cannot be found in the ones currently available, you can use the `keybert.backend.BaseEmbedder` class to
create your own backend. Below, you will find an example of creating a SentenceTransformer backend for KeyBERT:

```python
from keybert.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer

class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def embed(self, documents, verbose=False):
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings

# Create custom backend
distilbert = SentenceTransformer("paraphrase-MiniLM-L6-v2")
custom_embedder = CustomEmbedder(embedding_model=distilbert)

# Pass custom backend to keybert
kw_model = KeyBERT(model=custom_embedder)
```
