# import pytest
# import numpy as np
# import pandas as pd
# from unittest import mock
#
# from sklearn.datasets import fetch_20newsgroups, make_blobs
# from keybert import KeyBERT
#
# newsgroup_docs = fetch_20newsgroups(subset='all')['data'][:1000]
#
# @mock.patch("bertopic.model.BERTopic._extract_embeddings")
# def test_fit_transform(embeddings, base_bertopic):
#     """ Test whether predictions are correctly made """
#     blobs, _ = make_blobs(n_samples=len(newsgroup_docs), centers=5, n_features=768, random_state=42)
#     embeddings.return_value = blobs
#     predictions = base_bertopic.fit_transform(newsgroup_docs)
#
#     assert isinstance(predictions, list)
#     assert len(predictions) == len(newsgroup_docs)
#     assert not set(predictions).difference(set(base_bertopic.get_topics().keys()))
