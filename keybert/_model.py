# ruff: noqa: E402

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
from keybert._highlight import highlight_document
from keybert.backend._utils import select_backend
from keybert.llm._base import BaseLLM
from keybert import KeyLLM


class KeyBERT:
    """A minimal method for keyword extraction with BERT.

    The keyword extraction is done by finding the sub-phrases in
    a document that are the most similar to the document itself.

    First, document embeddings are extracted with BERT to get a
    document-level representation. Then, word embeddings are extracted
    for N-gram words/phrases. Finally, we use cosine similarity to find the
    words/phrases that are the most similar to the document.

    The most similar words could then be identified as the words that
    best describe the entire document.

    <div class="excalidraw">
    --8<-- "docs/images/pipeline.svg"
    </div>
    """

    def __init__(
        self,
        model="all-MiniLM-L6-v2",
        llm: BaseLLM = None,
    ):
        """KeyBERT initialization.

        Arguments:
            model: Use a custom embedding model or a specific KeyBERT Backend.
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
            llm: The Large Language Model used to extract keywords
        """
        self.model = select_backend(model)

        if isinstance(llm, BaseLLM):
            self.llm = KeyLLM(llm)
        else:
            self.llm = llm

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
        seed_keywords: Union[List[str], List[List[str]]] = None,
        doc_embeddings: np.array = None,
        word_embeddings: np.array = None,
        threshold: float = None,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Extract keywords and/or keyphrases.

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
                           NOTE: when multiple documents are passed,
                           `seed_keywords`funtions in either of the two ways below:
                           - globally: when a flat list of str is passed, keywords are shared by all documents,
                           - locally: when a nested list of str is passed, keywords differs among documents.
            doc_embeddings: The embeddings of each document.
            word_embeddings: The embeddings of each potential keyword/keyphrase across
                             across the vocabulary of the set of input documents.
                             NOTE: The `word_embeddings` should be generated through
                             `.extract_embeddings` as the order of these embeddings depend
                             on the vectorizer that was used to generate its vocabulary.
            threshold: Minimum similarity value between 0 and 1 used to decide how similar documents need to receive the same keywords.

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

        To extract keywords from multiple documents, which is typically quite a bit faster:

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

        # Check if the right number of word embeddings are generated compared with the vectorizer
        if word_embeddings is not None:
            if word_embeddings.shape[0] != len(words):
                raise ValueError(
                    "Make sure that the `word_embeddings` are generated from the function "
                    "`.extract_embeddings`. \nMoreover, the `candidates`, `keyphrase_ngram_range`,"
                    "`stop_words`, and `min_df` parameters need to have the same values in both "
                    "`.extract_embeddings` and `.extract_keywords`."
                )

        # Extract embeddings
        if doc_embeddings is None:
            doc_embeddings = self.model.embed(docs)
        if word_embeddings is None:
            word_embeddings = self.model.embed(words)

        # Guided KeyBERT either local (keywords shared among documents) or global (keywords per document)
        if seed_keywords is not None:
            if isinstance(seed_keywords[0], str):
                seed_embeddings = self.model.embed(seed_keywords).mean(axis=0, keepdims=True)
            elif len(docs) != len(seed_keywords):
                raise ValueError("The length of docs must match the length of seed_keywords")
            else:
                seed_embeddings = np.vstack(
                    [self.model.embed(keywords).mean(axis=0, keepdims=True) for keywords in seed_keywords]
                )
            doc_embeddings = (doc_embeddings * 3 + seed_embeddings) / 4

        # Find keywords
        all_keywords = []
        for index, _ in enumerate(docs):
            try:
                # Select embeddings
                candidate_indices = df[index].nonzero()[1]
                candidates = [words[index] for index in candidate_indices]
                candidate_embeddings = word_embeddings[candidate_indices]
                doc_embedding = doc_embeddings[index].reshape(1, -1)

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
            if highlight:
                highlight_document(docs[0], all_keywords[0], count)
            all_keywords = all_keywords[0]

        # Fine-tune keywords using an LLM
        if self.llm is not None:
            import torch

            doc_embeddings = torch.from_numpy(doc_embeddings).float()
            if torch.cuda.is_available():
                doc_embeddings = doc_embeddings.to("cuda")
            if isinstance(all_keywords[0], tuple):
                candidate_keywords = [[keyword[0] for keyword in all_keywords]]
            else:
                candidate_keywords = [[keyword[0] for keyword in keywords] for keywords in all_keywords]
            keywords = self.llm.extract_keywords(
                docs,
                embeddings=doc_embeddings,
                candidate_keywords=candidate_keywords,
                threshold=threshold,
            )
            return keywords
        return all_keywords

    def extract_embeddings(
        self,
        docs: Union[str, List[str]],
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        min_df: int = 1,
        vectorizer: CountVectorizer = None,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Extract document and word embeddings for the input documents and the
        generated candidate keywords/keyphrases respectively.

        Note that all potential keywords/keyphrases are not returned but only their
        word embeddings. This means that the values of `candidates`, `keyphrase_ngram_range`,
        `stop_words`, and `min_df` need to be the same between using `.extract_embeddings` and
        `.extract_keywords`.

        Arguments:
            docs: The document(s) for which to extract keywords/keyphrases
            candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)
                        NOTE: This is not used if you passed a `vectorizer`.
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases.
                                   NOTE: This is not used if you passed a `vectorizer`.
            stop_words: Stopwords to remove from the document.
                        NOTE: This is not used if you passed a `vectorizer`.
            min_df: Minimum document frequency of a word across all documents
                    if keywords for multiple documents need to be extracted.
                    NOTE: This is not used if you passed a `vectorizer`.
            vectorizer: Pass in your own `CountVectorizer` from
                        `sklearn.feature_extraction.text.CountVectorizer`

        Returns:
            doc_embeddings: The embeddings of each document.
            word_embeddings: The embeddings of each potential keyword/keyphrase across
                             across the vocabulary of the set of input documents.
                             NOTE: The `word_embeddings` should be generated through
                             `.extract_embeddings` as the order of these embeddings depend
                             on the vectorizer that was used to generate its vocabulary.

        Usage:

        To generate the word and document embeddings from a set of documents:

        ```python
        from keybert import KeyBERT

        kw_model = KeyBERT()
        doc_embeddings, word_embeddings = kw_model.extract_embeddings(docs)
        ```

        You can then use these embeddings and pass them to `.extract_keywords` to speed up the tuning the model:

        ```python
        keywords = kw_model.extract_keywords(docs, doc_embeddings=doc_embeddings, word_embeddings=word_embeddings)
        ```
        """
        # Check for a single, empty document
        if isinstance(docs, str):
            if docs:
                docs = [docs]
            else:
                return []

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

        doc_embeddings = self.model.embed(docs)
        word_embeddings = self.model.embed(words)

        return doc_embeddings, word_embeddings
