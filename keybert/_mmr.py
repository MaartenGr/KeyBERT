import numpy as np
from operator import itemgetter
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot


def mmr(
    doc_embedding: np.ndarray,
    word_embeddings: np.ndarray,
    words: List[str],
    top_n: int = 5,
    diversity: float = 0.8,
) -> List[Tuple[str, float]]:
    """Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.


    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.

    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyhprases to return
        diversity: How diverse the select keywords/keyphrases are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.

    Returns:
         List[Tuple[str, float]]: The selected keywords/keyphrases with their distances

    """
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding).ravel()
    best = np.argmax(word_doc_similarity)
    keywords_idx = [best]
    count = min(top_n, len(words))

    if count > 1:
        normalized = normalize(word_embeddings)
        selected = np.zeros(len(words), dtype=bool)
        selected[best] = True
        # Keep only the largest similarity to any keyword selected so far.
        redundancy = np.full(len(words), -np.inf, dtype=normalized.dtype)
        for _ in range(1, count):
            similarity = safe_sparse_dot(normalized, normalized[best : best + 1].T, dense_output=True).ravel()
            np.maximum(redundancy, similarity, out=redundancy)
            scores = (1 - diversity) * word_doc_similarity - diversity * redundancy
            scores[selected] = -np.inf
            best = np.argmax(scores)
            keywords_idx.append(best)
            selected[best] = True

    # Extract and sort keywords in descending similarity
    keywords = [(words[idx], round(float(word_doc_similarity[idx]), 4)) for idx in keywords_idx]
    keywords = sorted(keywords, key=itemgetter(1), reverse=True)
    return keywords
