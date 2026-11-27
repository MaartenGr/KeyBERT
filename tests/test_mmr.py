import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from keybert import KeyBERT
from keybert._mmr import mmr
from keybert.backend import BaseEmbedder


def reference(doc, embeddings, words, top_n, diversity):
    relevance = cosine_similarity(embeddings, doc).ravel()
    pairwise = cosine_similarity(embeddings)
    chosen = [np.argmax(relevance)]
    while len(chosen) < min(top_n, len(words)):
        remaining = [i for i in range(len(words)) if i not in chosen]
        redundancy = pairwise[np.ix_(remaining, chosen)].max(axis=1)
        scores = (1 - diversity) * relevance[remaining] - diversity * redundancy
        chosen.append(remaining[np.argmax(scores)])
    return sorted(
        [(words[i], round(float(relevance[i]), 4)) for i in chosen],
        key=lambda item: item[1],
        reverse=True,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("top_n", [1, 5, 35])
@pytest.mark.parametrize("diversity", [0, 0.5, 1])
def test_mmr_matches_pairwise_reference(dtype, sparse, top_n, diversity):
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(30, 16)).astype(dtype)
    embeddings[1] = embeddings[0]
    embeddings[2] = 0
    if sparse:
        embeddings = csr_matrix(embeddings)
    doc = rng.normal(size=(1, 16)).astype(dtype)
    words = [str(i) for i in range(30)]
    assert mmr(doc, embeddings, words, top_n, diversity) == reference(doc, embeddings, words, top_n, diversity)


def test_extract_keywords_mmr_precomputed():
    words = ["alpha", "bravo", "charlie"]
    doc = np.array([[1.0, 0.0]])
    embeddings = np.array([[1.0, 0.0], [0.99, 0.01], [0.8, 0.6]])
    result = KeyBERT(model=BaseEmbedder()).extract_keywords(
        "alpha bravo charlie",
        candidates=words,
        doc_embeddings=doc,
        word_embeddings=embeddings,
        use_mmr=True,
        diversity=0.8,
        top_n=2,
    )
    assert result == [("alpha", 1.0), ("charlie", 0.8)]
