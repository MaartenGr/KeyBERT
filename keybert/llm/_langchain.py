from typing import List

from tqdm import tqdm

from keybert.llm._base import BaseLLM
from keybert.llm._utils import process_candidate_keywords

DEFAULT_PROMPT_TEMPLATE = """
# Task
You are provided with a document and a list of candidate keywords.
Your task to is extract keywords from the document.
Use the candidate keywords to guide your extraction.

# Document
{DOCUMENT}

# Candidate Keywords
{CANDIDATES}


Now extract the keywords from the document.
The keywords must be comma separated .
"""


class LangChain(BaseLLM):
    """Using chains in langchain to generate keywords.


    NOTE: The resulting keywords are expected to a list of comma-sparated str so
    any changes to the prompt will have to ensure the foramt.

    Arguments:
        chain: A langchain chain that has two input parameters, `input_documents` and `query`.
        verbose: Set this to True if you want to see a progress bar for the
                 keyword extraction.

    Usage:

    To use this, you will need to install the langchain package first.
    Additionally, you will need an underlying LLM to support langchain,
    like openai:

    `pip install langchain`
    `pip install langchain-openai`

    Then, you can create your chain as follows:

    ```python
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    _llm = ChatOpenAI(
        model="gpt-4o",
        api_key="my-openai-api-key",
        temperature=0,
    )
    _prompt = PromptTemplate.from_template(LangChain.DEFAULT_PROMPT_TEMPLATE)  # the default prompt from KeyBERT
    chain = _prompt | _llm | StrOutputParser()
    ```

    Finally, you can pass the chain to KeyBERT as follows:

    ```python
    from keybert.llm import LangChain
    from keybert import KeyLLM

    # Create your LLM
    llm = LangChain(chain)

    # Load it in KeyLLM
    kw_model = KeyLLM(llm)

    # Extract keywords
    docs = [
        "KeyBERT: A minimal method for keyword extraction with BERT. The keyword extraction is done by finding the sub-phrases in a document that are the most similar to the document itself. First, document embeddings are extracted with BERT to get a document-level representation. Then, word embeddings are extracted for N-gram words/phrases. Finally, we use cosine similarity to find the words/phrases that are the most similar to the document. The most similar words could then be identified as the words that best describe the entire document.",
        "KeyLLM: A minimal method for keyword extraction with Large Language Models (LLM). The keyword extraction is done by simply asking the LLM to extract a number of keywords from a single piece of text.",
    ]
    candidates = [
        ["keyword extraction", "Large Language Models", "LLM", "BERT", "transformer", "embeddings"],
        ["keyword extraction", "Large Language Models", "LLM", "BERT", "transformer", "embeddings"],
    ]
    keywords = kw_model.extract_keywords(docs=docs, candidate_keywords=candidates)
    print(keywords)
    # [['keyword extraction', 'BERT', 'embeddings'], ['keyword extraction', 'Large Language Models', 'LLM']]
    ```

    You can also use a custom prompt:

    ```python
    prompt = "What are these documents about? Please give a single label."
    llm = LangChain(chain, prompt=prompt)
    ```
    """

    DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE

    def __init__(
        self,
        chain,
        verbose: bool = False,
    ):
        self.chain = chain
        self.verbose = verbose

    def extract_keywords(self, documents: List[str], candidate_keywords: List[List[str]] = None):
        """Extract topics.

        Arguments:
            documents: The documents to extract keywords from
            candidate_keywords: A list of candidate keywords that the LLM will fine-tune
                        For example, it will create a nicer representation of
                        the candidate keywords, remove redundant keywords, or
                        shorten them depending on the input prompt.

        Returns:
            all_keywords: All keywords for each document
        """
        all_keywords = []
        candidate_keywords = process_candidate_keywords(documents, candidate_keywords)

        for document, candidates in tqdm(zip(documents, candidate_keywords), disable=not self.verbose):
            keywords = self.chain.invoke({"DOCUMENT": document, "CANDIDATES": candidates})
            keywords = [keyword.strip() for keyword in keywords.split(",")]
            all_keywords.append(keywords)

        return all_keywords
