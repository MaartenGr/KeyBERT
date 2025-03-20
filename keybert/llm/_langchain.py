from typing import List

import langchain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel as LangChainBaseChatModel
from langchain_core.language_models.llms import BaseLLM as LangChainBaseLLM
from langchain_core.output_parsers import StrOutputParser
from packaging.version import InvalidVersion, Version
from tqdm import tqdm

from keybert.llm._base import BaseLLM
from keybert.llm._utils import process_candidate_keywords

if Version(langchain.__version__) < Version("0.1"):
    raise InvalidVersion(f"langchain>=0.1 is required, but langchain=={langchain.__version__} is installed.")

"""NOTE
KeyBERT only supports `langchain >= 0.1` which features:
- [Runnable Interface](https://python.langchain.com/docs/concepts/runnables/)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/concepts/lcel/)
"""


class LangChain(BaseLLM):
    """Using chains in langchain to generate keywords.

    Arguments:
        llm: A langchain LLM class. e.g ChatOpenAI, OpenAI, etc.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.DEFAULT_PROMPT_TEMPLATE` is used instead.
                NOTE: The prompt should contain:
                1. Placeholders
                - `[DOCUMENT]`: Required. The document to extract keywords from.
                - `[CANDIDATES]`: Optional. The candidate keywords to fine-tune the extraction.
                2. Output format instructions
                - Include this or something similar in your prompt:
                    "Extracted keywords must be separated by comma."
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
    from langchain_openai import ChatOpenAI

    _llm = ChatOpenAI(
        model="gpt-4o",
        api_key="my-openai-api-key",
        temperature=0,
    )
    ```

    Finally, you can pass the chain to KeyBERT as follows:

    ```python
    from keybert.llm import LangChain
    from keybert import KeyLLM

    # Create your LLM
    llm = LangChain(_llm)

    # Load it in KeyLLM
    kw_model = KeyLLM(llm)

    # Extract keywords
    docs = [
        "KeyBERT: A minimal method for keyword extraction with BERT. The keyword extraction is done by finding the sub-phrases in a document that are the most similar to the document itself. First, document embeddings are extracted with BERT to get a document-level representation. Then, word embeddings are extracted for N-gram words/phrases. Finally, we use cosine similarity to find the words/phrases that are the most similar to the document. The most similar words could then be identified as the words that best describe the entire document.",
        "KeyLLM: A minimal method for keyword extraction with Large Language Models (LLM). The keyword extraction is done by simply asking the LLM to extract a number of keywords from a single piece of text.",
    ]
    keywords = kw_model.extract_keywords(docs=docs)
    print(keywords)

    # Output:
    # [
    #     ['KeyBERT', 'keyword extraction', 'BERT', 'document embeddings', 'word embeddings', 'N-gram phrases', 'cosine similarity', 'document representation'],
    #     ['KeyLLM', 'keyword extraction', 'Large Language Models', 'LLM', 'minimal method']
    # ]


    # fine tune with candidate keywords
    candidates = [
        ["keyword extraction", "Large Language Models", "LLM", "BERT", "transformer", "embeddings"],
        ["keyword extraction", "Large Language Models", "LLM", "BERT", "transformer", "embeddings"],
    ]
    keywords = kw_model.extract_keywords(docs=docs, candidate_keywords=candidates)
    print(keywords)

    # Output:
    # [
    #     ['keyword extraction', 'BERT', 'document embeddings', 'word embeddings', 'cosine similarity', 'N-gram phrases'],
    #     ['KeyLLM', 'keyword extraction', 'Large Language Models', 'LLM']
    # ]
    ```

    You can also use a custom prompt:

    ```python
    prompt = "What are these documents about? Please give a single label."
    llm = LangChain(chain, prompt=prompt)
    ```
    """

    DEFAULT_PROMPT_TEMPLATE = """
# Task
You are provided with a document and possiblily a list of candidate keywords.

If no candidate keywords are provided, your task to is extract keywords from the document.
If candidate keywords are provided, your task is  to improve the candidate keywords to best describe the topic of the document.

# Document
[DOCUMENT]

# Candidate Keywords
[CANDIDATES]


Now extract the keywords from the document.
The keywords must be comma separated.
For example: "keyword1, keyword2, keyword3"
"""

    def __init__(
        self,
        llm: LangChainBaseChatModel | LangChainBaseLLM,
        prompt: str = None,
        verbose: bool = False,
    ):
        self.llm = llm
        self.prompt = prompt if prompt is not None else self.DEFAULT_PROMPT_TEMPLATE
        self.verbose = verbose
        self.chain = self._get_chain()

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

    def _get_chain(self):
        """Get the chain using LLM and prompt."""
        # format prompt for langchain template placeholders
        prompt = self.prompt.replace("[DOCUMENT]", "{DOCUMENT}").replace("[CANDIDATES]", "{CANDIDATES}")
        # check if the model is a chat model
        is_chat_model = isinstance(self.llm, LangChainBaseChatModel)
        # langchain prompt template
        prompt_template = ChatPromptTemplate([("human", prompt)]) if is_chat_model else PromptTemplate(template=prompt)
        # chain
        return prompt_template | self.llm | StrOutputParser()
