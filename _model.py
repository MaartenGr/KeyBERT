import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from typing import List, Union, Tuple

from packaging import version
from sklearn import __version__ as sklearn_version
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from keybert._mmr import mmr
from keybert._maxsum import max_sum_distance
from keybert.backend._utils import select_backend


class KeyBERT:
    """
    A minimal method for keyword extraction with BERT

    The keyword extraction is done by finding the sub-phrases in
    a document that are the most similar to the document itself.

    First, document embeddings are extracted with BERT to get a
    document-level representation. Then, word embeddings are extracted
    for N-gram words/phrases. Finally, we use cosine similarity to find the
    words/phrases that are the most similar to the document.

    The most similar words could then be identified as the words that
    best describe the entire document.
    """

    def __init__(self, model="all-MiniLM-L6-v2"):
        """KeyBERT initialization

        Arguments:
            model: Use a custom embedding model.
                   The following backends are currently supported:
                      * SentenceTransformers
                      * 🤗 Transformers
                      * Flair
                      * Spacy
                      * Gensim
                      * USE (TF-Hub)
                    You can also pass in a string that points to one of the following
                    sentence-transformers models:
                      * https://www.sbert.net/docs/pretrained_models.html
        """
        self.model = select_backend(model)

    def extract_keywords(
        self,
        docs: Union[str, List[str]],
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        min_df: int = 1,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
        vectorizer: CountVectorizer = None,
        seed_keywords: List[str] = None,
        doc_embeddings: np.ndarray = None
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Extract keywords and/or keyphrases

        To get the biggest speed-up, make sure to pass multiple documents
        at once instead of iterating over a single document.

        Arguments:
            docs: The document(s) for which to extract keywords/keyphrases
            candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)
                        NOTE: This is not used if you passed a `vectorizer`.
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases.
                                   NOTE: This is not used if you passed a `vectorizer`.
            stop_words: Stopwords to remove from the document.
                        NOTE: This is not used if you passed a `vectorizer`.
            top_n: Return the top n keywords/keyphrases
            min_df: Minimum document frequency of a word across all documents
                    if keywords for multiple documents need to be extracted.
                    NOTE: This is not used if you passed a `vectorizer`.
            use_maxsum: Whether to use Max Sum Distance for the selection
                        of keywords/keyphrases.
            use_mmr: Whether to use Maximal Marginal Relevance (MMR) for the
                     selection of keywords/keyphrases.
            diversity: The diversity of the results between 0 and 1 if `use_mmr`
                       is set to True.
            nr_candidates: The number of candidates to consider if `use_maxsum` is
                           set to True.
            vectorizer: Pass in your own `CountVectorizer` from
                        `sklearn.feature_extraction.text.CountVectorizer`
            highlight: Whether to print the document and highlight its keywords/keyphrases.
                       NOTE: This does not work if multiple documents are passed.
            seed_keywords: Seed keywords that may guide the extraction of keywords by
                           steering the similarities towards the seeded keywords.

        Returns:
            keywords: The top n keywords for a document with their respective distances
                      to the input document.

        Usage:

        To extract keywords from a single document:

        ```python
        from keybert import KeyBERT

        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(doc)
        ```

        To extract keywords from multiple documents,
        which is typically quite a bit faster:

        ```python
        from keybert import KeyBERT

        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(docs)
        ```
        """
        # Check for a single, empty document
        if isinstance(docs, str):
            if docs:
                docs = [docs]
            else:
                return []

        if candidates:
            word_embeddings = self.model.embed(candidates)
        else:
            # Extract potential words using a vectorizer / tokenizer
            if vectorizer:
                count = vectorizer.fit(docs)
            else:
                try:
                    count = CountVectorizer(
                        ngram_range=keyphrase_ngram_range,
                        stop_words=stop_words,
                        min_df=min_df,
                        vocabulary=candidates,
                    ).fit(docs)
                except ValueError:
                    return []

            # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
            # and will be removed in 1.2. Please use get_feature_names_out instead.
            if version.parse(sklearn_version) >= version.parse("1.0.0"):
                words = count.get_feature_names_out()
            else:
                words = count.get_feature_names()
            df = count.transform(docs)
            word_embeddings = self.model.embed(words)

        # Extract embeddings
        if doc_embeddings is None:
            doc_embeddings = self.model.embed(docs)

        # Find keywords
        all_keywords = []
        for index, _ in enumerate(docs):

            try:
                if candidates:
                    candidate_embeddings = word_embeddings
                else:
                    # Select embeddings
                    candidate_indices = df[index].nonzero()[1]
                    candidates = [words[index] for index in candidate_indices]
                    candidate_embeddings = word_embeddings[candidate_indices]
                doc_embedding = doc_embeddings[index].reshape(1, -1)

                # Guided KeyBERT with seed keywords
                if seed_keywords is not None:
                    seed_embeddings = self.model.embed([" ".join(seed_keywords)])
                    doc_embedding = np.average(
                        [doc_embedding, seed_embeddings], axis=0, weights=[3, 1]
                    )

                # Maximal Marginal Relevance (MMR)
                if use_mmr:
                    keywords = mmr(
                        doc_embedding,
                        candidate_embeddings,
                        candidates,
                        top_n,
                        diversity,
                    )

                # Max Sum Distance
                elif use_maxsum:
                    keywords = max_sum_distance(
                        doc_embedding,
                        candidate_embeddings,
                        candidates,
                        top_n,
                        nr_candidates,
                    )

                # Cosine-based keyword extraction
                else:
                    distances = cosine_similarity(doc_embedding, candidate_embeddings)
                    keywords = [
                        (candidates[index], round(float(distances[0][index]), 4))
                        for index in distances.argsort()[0][-top_n:]
                    ][::-1]

                all_keywords.append(keywords)

            # Capturing empty keywords
            except ValueError:
                all_keywords.append([])

        # Highlight keywords in the document
        if len(all_keywords) == 1:
            all_keywords = all_keywords[0]

        return all_keywords
