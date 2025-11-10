import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


def max_sum_distance(
    doc_embedding: np.ndarray,
    word_embeddings: np.ndarray,
    words: List[str],
    top_n: int,
    nr_candidates: int,
) -> List[Tuple[str, float]]:
    """Calculate Max Sum Distance for extraction of keywords.

    We take the 2 x top_n words/phrases most similar to the document.
    Among these words, we evaluate all possible `top_n`-combinations
    and select one combination where words are least similar to each other
    by cosine similarity.

    This has super-exponential time complexity (O(n^n)) and therefore
    not advised if you use a large `top_n`. For practical use, keep `top_n small
    (e.g., <= 10).

    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyhprases to return
        nr_candidates: The number of candidates to consider

    Returns:
         List[Tuple[str, float]]: The selected keywords/keyphrases with their distances
    """
    if nr_candidates < top_n:
        raise Exception("Make sure that the number of candidates exceeds the number of keywords to return.")
    elif top_n > len(words):
        return []

    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, word_embeddings)
    distances_words = cosine_similarity(word_embeddings, word_embeddings)

    # Get 2*top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [words[index] for index in words_idx]
    candidates = distances_words[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = 100_000
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [(words_vals[idx], round(float(distances[0][words_idx[idx]]), 4)) for idx in candidate]
