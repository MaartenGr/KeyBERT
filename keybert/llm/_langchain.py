from tqdm import tqdm
from typing import List
from langchain.docstore.document import Document
from keybert.llm._base import BaseLLM
from keybert.llm._utils import process_candidate_keywords


DEFAULT_PROMPT = "What is this document about? Please provide keywords separated by commas."


class LangChain(BaseLLM):
    """Using chains in langchain to generate keywords.

    Currently, only chains from question answering is implemented. See:
    https://langchain.readthedocs.io/en/latest/modules/chains/combine_docs_examples/question_answering.html

    NOTE: The resulting keywords are expected to be separated by commas so
    any changes to the prompt will have to make sure that the resulting
    keywords are comma-separated.

    Arguments:
        chain: A langchain chain that has two input parameters, `input_documents` and `query`.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
        verbose: Set this to True if you want to see a progress bar for the
                 keyword extraction.

    Usage:

    To use this, you will need to install the langchain package first.
    Additionally, you will need an underlying LLM to support langchain,
    like openai:

    `pip install langchain`
    `pip install openai`

    Then, you can create your chain as follows:

    ```python
    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI
    chain = load_qa_chain(OpenAI(temperature=0, openai_api_key=my_openai_api_key), chain_type="stuff")
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
    document = "The website mentions that it only takes a couple of days to deliver but I still have not received mine."
    keywords = kw_model.extract_keywords(document)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "What are these documents about? Please give a single label."
    llm = LangChain(chain, prompt=prompt)
    ```
    """

    def __init__(
        self,
        chain,
        prompt: str = None,
        verbose: bool = False,
    ):
        self.chain = chain
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT
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
            prompt = self.prompt.replace("[DOCUMENT]", document)
            if candidates is not None:
                prompt = prompt.replace("[CANDIDATES]", ", ".join(candidates))
            input_document = Document(page_content=document)
            keywords = self.chain.run(input_documents=[input_document], question=self.prompt).strip()
            keywords = [keyword.strip() for keyword in keywords.split(",")]
            all_keywords.append(keywords)

        return all_keywords
