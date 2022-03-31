import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# KeyBERT
from keybert._mmr import mmr
from keybert._maxsum import max_sum_similarity
from keybert._highlight import highlight_document
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
        highlight: bool = False,
        seed_keywords: List[str] = None,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Extract keywords and/or keyphrases

        I would advise you to iterate over single documents as they
        will need the least amount of memory. Even though this is slower,
        you are not likely to run into memory errors.

        There is an option to extract keywords for multiple documents
        that is faster than extraction for multiple single documents.
        However, this method assumes that you can keep the word embeddings
        for all words in the vocabulary in memory which might be troublesome.
        I would advise against using this option and simply iterating
        over documents instead if you have limited hardware.

        Arguments:
            docs: The document(s) for which to extract keywords/keyphrases
            candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases
            stop_words: Stopwords to remove from the document
            top_n: Return the top n keywords/keyphrases
            min_df: Minimum document frequency of a word across all documents
                    if keywords for multiple documents need to be extracted
            use_maxsum: Whether to use Max Sum Similarity for the selection
                        of keywords/keyphrases
            use_mmr: Whether to use Maximal Marginal Relevance (MMR) for the
                     selection of keywords/keyphrases
            diversity: The diversity of the results between 0 and 1 if use_mmr
                       is set to True
            nr_candidates: The number of candidates to consider if use_maxsum is
                           set to True
            vectorizer: Pass in your own CountVectorizer from scikit-learn
            highlight: Whether to print the document and highlight
                       its keywords/keyphrases. NOTE: This does not work if
                       multiple documents are passed.
            seed_keywords: Seed keywords that may guide the extraction of keywords by
                           steering the similarities towards the seeded keywords

        Returns:
            keywords: the top n keywords for a document with their respective distances
                      to the input document

        """

        if isinstance(docs, str):
            keywords = self._extract_keywords_single_doc(
                doc=docs,
                candidates=candidates,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words=stop_words,
                top_n=top_n,
                use_maxsum=use_maxsum,
                use_mmr=use_mmr,
                diversity=diversity,
                nr_candidates=nr_candidates,
                vectorizer=vectorizer,
                seed_keywords=seed_keywords,
            )
            if highlight:
                highlight_document(docs, keywords)

            return keywords

        elif isinstance(docs, list):
            warnings.warn(
                "Although extracting keywords for multiple documents is faster "
                "than iterating over single documents, it requires significantly more memory "
                "to hold all word embeddings. Use this at your own discretion!"
            )
            return self._extract_keywords_multiple_docs(
                docs, keyphrase_ngram_range, stop_words, top_n, min_df, vectorizer
            )

    def _extract_keywords_single_doc(
        self,
        doc: str,
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
        vectorizer: CountVectorizer = None,
        seed_keywords: List[str] = None,
    ) -> List[Tuple[str, float]]:
        """Extract keywords/keyphrases for a single document

        Arguments:
            doc: The document for which to extract keywords/keyphrases
            candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases
            stop_words: Stopwords to remove from the document
            top_n: Return the top n keywords/keyphrases
            use_mmr: Whether to use Max Sum Similarity
            use_mmr: Whether to use MMR
            diversity: The diversity of results between 0 and 1 if use_mmr is True
            nr_candidates: The number of candidates to consider if use_maxsum is set to True
            vectorizer: Pass in your own CountVectorizer from scikit-learn
            seed_keywords: Seed keywords that may guide the extraction of keywords by
                           steering the similarities towards the seeded keywords

        Returns:
            keywords: the top n keywords for a document with their respective distances
                      to the input document
        """
        try:
            # Extract Words
            if candidates is None:
                if vectorizer:
                    count = vectorizer.fit([doc])
                else:
                    count = CountVectorizer(
                        ngram_range=keyphrase_ngram_range, stop_words=stop_words
                    ).fit([doc])
                candidates = count.get_feature_names()

            # Extract Embeddings
            doc_embedding = self.model.embed([doc])
            candidate_embeddings = self.model.embed(candidates)

            # Guided KeyBERT with seed keywords
            if seed_keywords is not None:
                seed_embeddings = self.model.embed([" ".join(seed_keywords)])
                doc_embedding = np.average(
                    [doc_embedding, seed_embeddings], axis=0, weights=[3, 1]
                )

            # Calculate distances and extract keywords
            if use_mmr:
                keywords = mmr(
                    doc_embedding, candidate_embeddings, candidates, top_n, diversity
                )
            elif use_maxsum:
                keywords = max_sum_similarity(
                    doc_embedding,
                    candidate_embeddings,
                    candidates,
                    top_n,
                    nr_candidates,
                )
            else:
                distances = cosine_similarity(doc_embedding, candidate_embeddings)
                keywords = [
                    (candidates[index], round(float(distances[0][index]), 4))
                    for index in distances.argsort()[0][-top_n:]
                ][::-1]

            return keywords
        except ValueError:
            return []

    def _extract_keywords_multiple_docs(
        self,
        docs: List[str],
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: str = "english",
        top_n: int = 5,
        min_df: int = 1,
        vectorizer: CountVectorizer = None,
    ) -> List[List[Tuple[str, float]]]:
        """Extract keywords/keyphrases for a multiple documents

        This currently does not use MMR and Max Sum Similarity as it cannot
        process these methods in bulk.

        Arguments:
            docs: The document for which to extract keywords/keyphrases
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases
            stop_words: Stopwords to remove from the document
            top_n: Return the top n keywords/keyphrases
            min_df: The minimum frequency of words
            vectorizer: Pass in your own CountVectorizer from scikit-learn

        Returns:
            keywords: the top n keywords for a document with their respective distances
                      to the input document
        """
        # Extract words
        if vectorizer:
            count = vectorizer.fit(docs)
        else:
            count = CountVectorizer(
                ngram_range=keyphrase_ngram_range, stop_words=stop_words, min_df=min_df
            ).fit(docs)
        words = count.get_feature_names()
        df = count.transform(docs)

        # Extract embeddings
        doc_embeddings = self.model.embed(docs)
        word_embeddings = self.model.embed(words)

        # Extract keywords
        keywords = []
        for index, doc in tqdm(enumerate(docs)):
            doc_words = [words[i] for i in df[index].nonzero()[1]]

            if doc_words:
                doc_word_embeddings = np.array(
                    [word_embeddings[i] for i in df[index].nonzero()[1]]
                )
                distances = cosine_similarity(
                    [doc_embeddings[index]], doc_word_embeddings
                )[0]
                doc_keywords = [
                    (doc_words[i], round(float(distances[i]), 4))
                    for i in distances.argsort()[-top_n:]
                ]
                keywords.append(doc_keywords)
            else:
                keywords.append(["None Found"])

        return keywords
