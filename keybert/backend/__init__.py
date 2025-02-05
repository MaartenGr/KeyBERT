from keybert.backend._base import BaseEmbedder
from keybert._utils import NotInstalled

# Sentence Transformers
try:
    from ._sentencetransformers import SentenceTransformerBackend
except ModuleNotFoundError:
    msg = "`pip install sentence-transformers`"
    SentenceTransformerBackend = NotInstalled("Sentence-Transformers", "sentence-transformers", custom_msg=msg)

# Model2Vec
try:
    from ._model2vec import Model2VecBackend
except ModuleNotFoundError:
    msg = "`pip install model2vec`"
    Model2VecBackend = NotInstalled("Model2Vec", "Model2Vec", custom_msg=msg)


__all__ = ["BaseEmbedder", "SentenceTransformerBackend", "Model2VecBackend"]
