import os
import time
from keybert._model import KeyBERT

os.environ['TORCH_HOME'] = './'

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
doc_embeddings = kw_model.model.embed([doc])

keywords = kw_model.extract_keywords(doc, doc_embeddings=doc_embeddings)
print(keywords)

keywords = kw_model.extract_keywords(doc, candidates=["input"], doc_embeddings=doc_embeddings)
print(keywords)
