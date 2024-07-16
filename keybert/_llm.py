from typing import List, Union

try:
    from sentence_transformers import util

    HAS_SBERT = True
except ModuleNotFoundError:
    HAS_SBERT = False


class KeyLLM:
    """A minimal method for keyword extraction with Large Language Models (LLM).

    The keyword extraction is done by simply asking the LLM to extract a
    number of keywords from a single piece of text.
    """

    def __init__(self, llm):
        """KeyBERT initialization.

        Arguments:
            llm: The Large Language Model to use
        """
        self.llm = llm

    def extract_keywords(
        self,
        docs: Union[str, List[str]],
        check_vocab: bool = False,
        candidate_keywords: List[List[str]] = None,
        threshold: float = None,
        embeddings=None,
    ) -> Union[List[str], List[List[str]]]:
        """Extract keywords and/or keyphrases.

        To get the biggest speed-up, make sure to pass multiple documents
        at once instead of iterating over a single document.

        NOTE: The resulting keywords are expected to be separated by commas so
        any changes to the prompt will have to make sure that the resulting
        keywords are comma-separated.

        Arguments:
            docs: The document(s) for which to extract keywords/keyphrases
            check_vocab: Only return keywords that appear exactly in the documents
            candidate_keywords: Candidate keywords for each document
            threshold: Minimum similarity value between 0 and 1 used to decide how similar documents need to receive the same keywords.
            embeddings: The embeddings of each document.

        Returns:
            keywords: The top n keywords for a document with their respective distances
                      to the input document.

        Usage:

        To extract keywords from a single document:

        ```python
        import openai
        from keybert.llm import OpenAI
        from keybert import KeyLLM

        # Create your LLM
        client = openai.OpenAI(api_key=MY_API_KEY)
        llm = OpenAI(client)

        # Load it in KeyLLM
        kw_model = KeyLLM(llm)

        # Extract keywords
        document = "The website mentions that it only takes a couple of days to deliver but I still have not received mine."
        keywords = kw_model.extract_keywords(document)
        ```
        """
        # Check for a single, empty document
        if isinstance(docs, str):
            if docs:
                docs = [docs]
            else:
                return []

        if HAS_SBERT and threshold is not None and embeddings is not None:
            # Find similar documents
            clusters = util.community_detection(embeddings, min_community_size=2, threshold=threshold)
            in_cluster = set([cluster for cluster_set in clusters for cluster in cluster_set])
            out_cluster = set(list(range(len(docs)))).difference(in_cluster)

            # Extract keywords for all documents not in a cluster
            if out_cluster:
                selected_docs = [docs[index] for index in out_cluster]
                if candidate_keywords is not None:
                    selected_keywords = [candidate_keywords[index] for index in out_cluster]
                else:
                    selected_keywords = None
                out_cluster_keywords = self.llm.extract_keywords(
                    selected_docs,
                    selected_keywords,
                )
                out_cluster_keywords = {index: words for words, index in zip(out_cluster_keywords, out_cluster)}

            # Extract keywords for only the first document in a cluster
            if in_cluster:
                selected_docs = [docs[cluster[0]] for cluster in clusters]
                if candidate_keywords is not None:
                    selected_keywords = [candidate_keywords[cluster[0]] for cluster in clusters]
                else:
                    selected_keywords = None
                in_cluster_keywords = self.llm.extract_keywords(selected_docs, selected_keywords)
                in_cluster_keywords = {
                    doc_id: in_cluster_keywords[index] for index, cluster in enumerate(clusters) for doc_id in cluster
                }

            # Update out cluster keywords with in cluster keywords
            if out_cluster:
                if in_cluster:
                    out_cluster_keywords.update(in_cluster_keywords)
                keywords = [out_cluster_keywords[index] for index in range(len(docs))]
            else:
                keywords = [in_cluster_keywords[index] for index in range(len(docs))]
        else:
            # Extract keywords using a Large Language Model (LLM)
            keywords = self.llm.extract_keywords(docs, candidate_keywords)

        # Only extract keywords that appear in the input document
        if check_vocab:
            updated_keywords = []
            for keyword_set, document in zip(keywords, docs):
                updated_keyword_set = []
                for keyword in keyword_set:
                    if keyword in document:
                        updated_keyword_set.append(keyword)
                updated_keywords.append(updated_keyword_set)
            return updated_keywords

        return keywords
