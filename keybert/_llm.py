from typing import List, Union


class KeyLLM:
    """
    A minimal method for keyword extraction with Large Language Models (LLM)

    The keyword extraction is done by simply asking the LLM to extract a
    number of keywords from a single piece of text.
    """

    def __init__(self, llm):
        """KeyBERT initialization

        Arguments:
            llm: The Large Language Model to use
        """
        self.llm = llm

    def extract_keywords(
        self,
        docs: Union[str, List[str]],
    ) -> Union[List[str], List[List[str]]]:
        """Extract keywords and/or keyphrases

        To get the biggest speed-up, make sure to pass multiple documents
        at once instead of iterating over a single document.

        NOTE: The resulting keywords are expected to be separated by commas so
        any changes to the prompt will have to make sure that the resulting
        keywords are comma-separated.

        Arguments:
            docs: The document(s) for which to extract keywords/keyphrases
            top_n: Return the top n keywords/keyphrases

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
        openai.api_key = "sk-..."
        llm = OpenAI()

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

        keywords = self.llm.extract_keywords(docs)
        return keywords
