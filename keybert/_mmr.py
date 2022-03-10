import numpy as np
from typing import List, Tuple

def mmr_fast(doc_embedding: np.ndarray,
            word_embeddings: np.ndarray,
            words: List[str],
            top_n: int = 5,
            diversity: float = 0.8) -> List[Tuple[str, float]]:
    """ Calculate Maximal Marginal Relevance (MMR)
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

    # Extract similarity within words, and between words and the document
    word_doc_similarity = calc_cos_sim_einsum(word_embeddings, doc_embedding).swapaxes(0,1)
    word_similarity = calc_cos_matrix(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(min(top_n - 1, len(words) - 1)):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [(words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4)) for idx in keywords_idx]

def calc_cos_sim_einsum(embeddings, target_emb):
    return np.dot(target_emb, embeddings.T)/(np.linalg.norm(target_emb)*np.sqrt(np.einsum('ij,ij->i',embeddings,embeddings)))

def calc_cos_matrix(embeddings):
    p = embeddings / np.linalg.norm(embeddings, 2, axis=1).reshape(-1,1)
    return np.dot(p, p.T)