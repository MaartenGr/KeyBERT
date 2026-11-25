import itertools

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from keybert import KeyBERT
from keybert import _maxsum
from keybert.backend import BaseEmbedder


def _reference(doc_embedding, word_embeddings, words, top_n, nr_candidates):
    distances = cosine_similarity(doc_embedding, word_embeddings)
    pairwise = cosine_similarity(word_embeddings, word_embeddings)
    indices = distances.argsort()[0][-nr_candidates:]
    candidates = pairwise[np.ix_(indices, indices)]
    best = min(
        itertools.combinations(range(len(indices)), top_n),
        key=lambda combination: sum(candidates[i, j] for i in combination for j in combination if i != j),
    )
    return [(words[indices[i]], round(float(distances[0, indices[i]]), 4)) for i in best]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nr_candidates", [3, 6, 20])
@pytest.mark.parametrize("top_n", [1, 3])
@pytest.mark.parametrize("seed", [0, 42, 123])
def test_maxsum_matches_full_matrix(dtype, nr_candidates, top_n, seed):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(12, 8)).astype(dtype)
    doc = rng.normal(size=(1, 8)).astype(dtype)
    words = [f"word{i}" for i in range(len(embeddings))]
    expected = _reference(doc, embeddings, words, top_n, nr_candidates)
    assert _maxsum.max_sum_distance(doc, embeddings, words, top_n, nr_candidates) == expected


@pytest.mark.parametrize("embeddings", [np.ones((6, 4)), np.zeros((6, 4)), np.eye(6)])
def test_maxsum_preserves_tie_order(embeddings):
    doc = np.ones((1, embeddings.shape[1]))
    words = [f"word{i}" for i in range(len(embeddings))]
    assert _maxsum.max_sum_distance(doc, embeddings, words, 2, 4) == _reference(doc, embeddings, words, 2, 4)


def test_maxsum_limits_pairwise_matrix_to_shortlist(monkeypatch):
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(1000, 16))
    doc = rng.normal(size=(1, 16))
    words = [f"word{i}" for i in range(len(embeddings))]
    shapes = []

    def tracked_similarity(x, y=None):
        shapes.append((len(x), len(x) if y is None else len(y)))
        return cosine_similarity(x, y)

    monkeypatch.setattr(_maxsum, "cosine_similarity", tracked_similarity)
    result = _maxsum.max_sum_distance(doc, embeddings, words, 3, 6)
    assert len(result) == 3
    assert shapes == [(1, 1000), (6, 6)]


def test_maxsum_with_precomputed_embeddings():
    model = KeyBERT(model=BaseEmbedder())
    words = ["alpha", "beta", "delta", "gamma"]
    embeddings = np.eye(4)
    doc = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert model.extract_keywords(
        "alpha beta delta gamma",
        stop_words=None,
        use_maxsum=True,
        top_n=2,
        nr_candidates=3,
        doc_embeddings=doc,
        word_embeddings=embeddings,
    ) == _reference(doc, embeddings, words, 2, 3)
