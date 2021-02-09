import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from .mmr import mmr
from .maxsum import max_sum_similarity

# Flair
try:
    from flair.embeddings import DocumentEmbeddings, TokenEmbeddings, DocumentPoolEmbeddings
    from flair.data import Sentence
    _HAS_FLAIR = True
except ModuleNotFoundError as e:
    DocumentEmbeddings, TokenEmbeddings, DocumentPoolEmbeddings = None, None, None
    _HAS_FLAIR = False


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
    def __init__(self,
                 model: Union[str,
                              SentenceTransformer,
                              DocumentEmbeddings,
                              TokenEmbeddings] = 'distilbert-base-nli-mean-tokens'):
        """ KeyBERT initialization

        Arguments:
            model: Use a custom embedding model. You can pass in a string related
                   to one of the following models:
                   https://www.sbert.net/docs/pretrained_models.html
                   You can also pass in a SentenceTransformer() model or a Flair
                   DocumentEmbedding model.
        """
        self.model = self._select_embedding_model(model)

    def extract_keywords(self,
                         docs: Union[str, List[str]],
                         keyphrase_ngram_range: Tuple[int, int] = (1, 1),
                         stop_words: Union[str, List[str]] = 'english',
                         top_n: int = 5,
                         min_df: int = 1,
                         use_maxsum: bool = False,
                         use_mmr: bool = False,
                         diversity: float = 0.5,
                         nr_candidates: int = 20,
                         vectorizer: CountVectorizer = None) -> Union[List[Tuple[str, float]],
                                                                      List[List[Tuple[str, float]]]]:
        """ Extract keywords/keyphrases

        NOTE:
            I would advise you to iterate over single documents as they
            will need the least amount of memory. Even though this is slower,
            you are not likely to run into memory errors.

        Multiple Documents:
            There is an option to extract keywords for multiple documents
            that is faster than extraction for multiple single documents.

            However...this method assumes that you can keep the word embeddings
            for all words in the vocabulary in memory which might be troublesome.

            I would advise against using this option and simply iterating
            over documents instead if you have limited hardware.

        Arguments:
            docs: The document(s) for which to extract keywords/keyphrases
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

        Returns:
            keywords: the top n keywords for a document with their respective distances
                      to the input document

        """

        if isinstance(docs, str):
            return self._extract_keywords_single_doc(docs,
                                                     keyphrase_ngram_range,
                                                     stop_words,
                                                     top_n,
                                                     use_maxsum,
                                                     use_mmr,
                                                     diversity,
                                                     nr_candidates,
                                                     vectorizer)
        elif isinstance(docs, list):
            warnings.warn("Although extracting keywords for multiple documents is faster "
                          "than iterating over single documents, it requires significantly more memory "
                          "to hold all word embeddings. Use this at your own discretion!")
            return self._extract_keywords_multiple_docs(docs,
                                                        keyphrase_ngram_range,
                                                        stop_words,
                                                        top_n,
                                                        min_df,
                                                        vectorizer)

    def _extract_keywords_single_doc(self,
                                     doc: str,
                                     keyphrase_ngram_range: Tuple[int, int] = (1, 1),
                                     stop_words: Union[str, List[str]] = 'english',
                                     top_n: int = 5,
                                     use_maxsum: bool = False,
                                     use_mmr: bool = False,
                                     diversity: float = 0.5,
                                     nr_candidates: int = 20,
                                     vectorizer: CountVectorizer = None) -> List[Tuple[str, float]]:
        """ Extract keywords/keyphrases for a single document

        Arguments:
            doc: The document for which to extract keywords/keyphrases
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases
            stop_words: Stopwords to remove from the document
            top_n: Return the top n keywords/keyphrases
            use_mmr: Whether to use Max Sum Similarity
            use_mmr: Whether to use MMR
            diversity: The diversity of results between 0 and 1 if use_mmr is True
            nr_candidates: The number of candidates to consider if use_maxsum is set to True
            vectorizer: Pass in your own CountVectorizer from scikit-learn

        Returns:
            keywords: the top n keywords for a document with their respective distances
                      to the input document

        """
        try:
            # Extract Words
            if vectorizer:
                count = vectorizer.fit([doc])
            else:
                count = CountVectorizer(ngram_range=keyphrase_ngram_range, stop_words=stop_words).fit([doc])
            words = count.get_feature_names()

            # Extract Embeddings
            doc_embedding = self._extract_embeddings([doc])
            word_embeddings = self._extract_embeddings(words)
            # doc_embedding = self.model.encode([doc])
            # word_embeddings = self.model.encode(words)

            # Calculate distances and extract keywords
            if use_mmr:
                keywords = mmr(doc_embedding, word_embeddings, words, top_n, diversity)
            elif use_maxsum:
                keywords = max_sum_similarity(doc_embedding, word_embeddings, words, top_n, nr_candidates)
            else:
                distances = cosine_similarity(doc_embedding, word_embeddings)
                keywords = [(words[index], round(float(distances[0][index]), 4))
                            for index in distances.argsort()[0][-top_n:]][::-1]

            return keywords
        except ValueError:
            return []

    def _extract_keywords_multiple_docs(self,
                                        docs: List[str],
                                        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
                                        stop_words: str = 'english',
                                        top_n: int = 5,
                                        min_df: int = 1,
                                        vectorizer: CountVectorizer = None) -> List[List[Tuple[str, float]]]:
        """ Extract keywords/keyphrases for a multiple documents

        This currently does not use MMR as

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
            count = CountVectorizer(ngram_range=keyphrase_ngram_range, stop_words=stop_words, min_df=min_df).fit(docs)
        words = count.get_feature_names()
        df = count.transform(docs)

        # Extract embeddings
        word_embeddings = self._extract_embeddings(words)
        doc_embeddings = self._extract_embeddings(docs)
        # word_embeddings = self.model.encode(words, show_progress_bar=True)
        # doc_embeddings = self.model.encode(docs, show_progress_bar=True)

        # Extract keywords
        keywords = []
        for index, doc in tqdm(enumerate(docs)):
            doc_words = [words[i] for i in df[index].nonzero()[1]]

            if doc_words:
                doc_word_embeddings = np.array([word_embeddings[i] for i in df[index].nonzero()[1]])
                distances = cosine_similarity([doc_embeddings[index]], doc_word_embeddings)[0]
                doc_keywords = [(doc_words[i], round(float(distances[i]), 4)) for i in distances.argsort()[-top_n:]]
                keywords.append(doc_keywords)
            else:
                keywords.append(["None Found"])

        return keywords

    def _extract_embeddings(self, documents: Union[List[str], str]) -> np.ndarray:
        """ Extract sentence/document embeddings through pre-trained embeddings

        For an overview of pre-trained models: https://www.sbert.net/docs/pretrained_models.html

        Arguments:
            documents: Dataframe with documents and their corresponding IDs

        Returns:
            embeddings: The extracted embeddings using the sentence transformer
                        module. Typically uses pre-trained huggingface models.
        """
        if isinstance(documents, str):
            documents = [documents]

        # Infer embeddings with SentenceTransformer
        if isinstance(self.model, SentenceTransformer):
            embeddings = self.model.encode(documents)

        # Infer embeddings with Flair
        elif isinstance(self.model, DocumentEmbeddings):
            embeddings = []
            for index, document in enumerate(documents):
                try:
                    sentence = Sentence(document) if document else Sentence("an empty document")
                    self.model.embed(sentence)
                except RuntimeError:
                    sentence = Sentence("an empty document")
                    self.model.embed(sentence)
                embedding = sentence.embedding.detach().cpu().numpy()
                embeddings.append(embedding)
            embeddings = np.asarray(embeddings)

        else:
            raise ValueError("An incorrect embedding model type was selected.")

        return embeddings

    def _select_embedding_model(self, model: Union[str,
                                                   SentenceTransformer,
                                                   DocumentEmbeddings,
                                                   TokenEmbeddings]) -> Union[SentenceTransformer,
                                                                              DocumentEmbeddings]:
        """ Select an embedding model based on language or a specific sentence transformer models.
        When selecting a language, we choose distilbert-base-nli-stsb-mean-tokens for English and
        xlm-r-bert-base-nli-stsb-mean-tokens for all other languages as it support 100+ languages.

        Arguments:
            model: Use a custom embedding model. You can pass in a string related
                   to one of the following models:
                   https://www.sbert.net/docs/pretrained_models.html
                   You can also pass in a SentenceTransformer() model or a Flair
                   DocumentEmbedding model.

        Returns:
            model: Either a Sentence-Transformer or Flair model
        """

        # Sentence Transformer embeddings
        if isinstance(model, SentenceTransformer):
            return model

        # Flair word embeddings
        elif _HAS_FLAIR and isinstance(model, TokenEmbeddings):
            return DocumentPoolEmbeddings([model])

        # Flair document embeddings + disable fine tune to prevent CUDA OOM
        # https://github.com/flairNLP/flair/issues/1719
        elif _HAS_FLAIR and isinstance(model, DocumentEmbeddings):
            if "fine_tune" in model.__dict__:
                model.fine_tune = False
            return model

        # Select embedding model based on specific sentence transformer model
        elif isinstance(model, str):
            return SentenceTransformer(model)

        return SentenceTransformer("xlm-r-bert-base-nli-stsb-mean-tokens")
