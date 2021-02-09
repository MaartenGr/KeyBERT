## **Embedding Models**
The parameter `model` takes in a string pointing to a sentence-transformers model, 
a SentenceTransformer, or a Flair DocumentEmbedding model. 

### **Sentence-Transformers**  
You can select any model from `sentence-transformers` [here](https://www.sbert.net/docs/pretrained_models.html) 
and pass it through KeyBERT with `model`:

```python
from keybert import KeyBERT
model = KeyBERT(model='distilbert-base-nli-mean-tokens')
```

Or select a SentenceTransformer model with your own parameters:

```python
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens", device="cpu")
model = KeyBERT(model=sentence_model)
```

### **Flair**  
[Flair](https://github.com/flairNLP/flair) allows you to choose almost any embedding model that 
is publicly available. Flair can be used as follows:

```python
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings

roberta = TransformerDocumentEmbeddings('roberta-base')
model = KeyBERT(model=roberta)
```

You can select any 🤗 transformers model [here](https://huggingface.co/models).
