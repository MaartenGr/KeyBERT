import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from typing import List, Union
import warnings


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

    Arguments:
        model: the name of the model used by sentence-transformer
               for a full overview see https://www.sbert.net/docs/pretrained_models.html

    """
    def __init__(self, model: str = 'distilbert-base-nli-mean-tokens'):
        self.model = SentenceTransformer(model)
        self.doc_embeddings = None

    def extract_keywords(self,
                         docs: Union[str, List[str]],
                         keyphrase_length: int = 1,
                         stop_words: Union[str, List[str]] = 'english',
                         top_n: int = 5,
                         min_df: int = 1) -> Union[List[str], List[List[str]]]:
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
            keyphrase_length: Length, in words, of the extracted keywords/keyphrases
            stop_words: Stopwords to remove from the document
            top_n: Return the top n keywords/keyphrases
            min_df: Minimum document frequency of a word across all documents
                    if keywords for multiple documents need to be extracted

        Returns:
            keywords: the top n keywords for a document

        """

        if isinstance(docs, str):
            return self._extract_keywords_single_doc(docs,
                                                     keyphrase_length,
                                                     stop_words,
                                                     top_n)
        elif isinstance(docs, list):
            warnings.warn("Although extracting keywords for multiple documents is faster "
                          "than iterating over single documents, it requires significant memory "
                          "to hold all word embeddings. Use this at your own discretion!")
            return self._extract_keywords_multiple_docs(docs,
                                                        keyphrase_length,
                                                        stop_words,
                                                        top_n,
                                                        min_df=min_df)

    def _extract_keywords_single_doc(self,
                                     doc: str,
                                     keyphrase_length: int = 1,
                                     stop_words: Union[str, List[str]] = 'english',
                                     top_n: int = 5) -> List[str]:
        """ Extract keywords/keyphrases for a single document

        Arguments:
            doc: The document for which to extract keywords/keyphrases
            keyphrase_length: Length, in words, of the extracted keywords/keyphrases
            stop_words: Stopwords to remove from the document
            top_n: Return the top n keywords/keyphrases

        Returns:
            keywords: The top n keywords for a document

        """
        try:
            # Extract Words
            n_gram_range = (keyphrase_length, keyphrase_length)
            count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
            words = count.get_feature_names()

            # Extract Embeddings
            doc_embeddings = self.model.encode([doc])
            word_embeddings = self.model.encode(words)

            # Calculate distances and extract keywords
            distances = cosine_similarity(doc_embeddings, word_embeddings)
            keywords = [words[index] for index in distances.argsort()[0][-top_n:]]

            return keywords[::-1]
        except ValueError:
            return []

    def _extract_keywords_multiple_docs(self,
                                        docs: List[str],
                                        keyphrase_length: int = 1,
                                        stop_words: str = 'english',
                                        top_n: int = 5,
                                        min_df: int = 1):
        """ Extract keywords/keyphrases for a multiple documents

        Arguments:
            docs: The document for which to extract keywords/keyphrases
            keyphrase_length: Length, in words, of the extracted keywords/keyphrases
            stop_words: Stopwords to remove from the document
            top_n: Return the top n keywords/keyphrases
            min_df: The minimum frequency of words

        Returns:
            keywords: The top n keywords for a document

        """
        # Extract words
        n_gram_range = (keyphrase_length, keyphrase_length)
        count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words, min_df=min_df).fit(docs)
        words = count.get_feature_names()
        df = count.transform(docs)

        # Extract embeddings
        word_embeddings = self.model.encode(words, show_progress_bar=True)
        doc_embeddings = self.model.encode(docs, show_progress_bar=True)

        # Extract keywords
        keywords = []
        for index, doc in tqdm(enumerate(docs)):
            doc_words = [words[i] for i in df[index].nonzero()[1]]

            if doc_words:
                doc_word_embeddings = np.array([word_embeddings[i] for i in df[index].nonzero()[1]])
                distances = cosine_similarity([doc_embeddings[index]], doc_word_embeddings)[0]
                doc_keywords = [doc_words[i] for i in distances.argsort()[-top_n:]]
                keywords.append(doc_keywords)
            else:
                keywords.append(["None Found"])

        return keywords
