import time
from tqdm import tqdm
from typing import List
from keybert.llm._base import BaseLLM
from keybert.llm._utils import process_candidate_keywords


DEFAULT_PROMPT = """
The following is a list of documents. Please extract the top keywords, separated by a comma, that describe the topic of the texts.

Document:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.

Keywords: Traditional diets, Plant-based, Meat, Industrial style meat production, Factory farming, Staple food, Cultural dietary practices

Document:
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.

Keywords: Website, Delivery, Mention, Timeframe, Not received, Waiting, Order fulfillment

Document:
- [DOCUMENT]

Keywords:"""


class Cohere(BaseLLM):
    """Use the Cohere API to generate topic labels based on their
    generative model.

    Find more about their models here:
    https://docs.cohere.ai/docs

    NOTE: The resulting keywords are expected to be separated by commas so
    any changes to the prompt will have to make sure that the resulting
    keywords are comma-separated.

    Arguments:
        client: A cohere.Client
        model: Model to use within Cohere, defaults to `"xlarge"`.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        delay_in_seconds: The delay in seconds between consecutive prompts
                          in order to prevent RateLimitErrors.
        verbose: Set this to True if you want to see a progress bar for the
                 keyword extraction.

    Usage:

    To use this, you will need to install cohere first:

    `pip install cohere`

    Then, get yourself an API key and use Cohere's API as follows:

    ```python
    import cohere
    from keybert.llm import Cohere
    from keybert import KeyLLM

    # Create your LLM
    co = cohere.Client(my_api_key)
    llm = Cohere(co)

    # Load it in KeyLLM
    kw_model = KeyLLM(llm)

    # Extract keywords
    document = "The website mentions that it only takes a couple of days to deliver but I still have not received mine."
    keywords = kw_model.extract_keywords(document)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following document: [DOCUMENT]. What keywords does it contain? Make sure to separate the keywords with commas."
    llm = Cohere(co, prompt=prompt)
    ```
    """

    def __init__(
        self, client, model: str = "command", prompt: str = None, delay_in_seconds: float = None, verbose: bool = False
    ):
        self.client = client
        self.model = model
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT
        self.delay_in_seconds = delay_in_seconds
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

            # Delay
            if self.delay_in_seconds:
                time.sleep(self.delay_in_seconds)

            request = self.client.generate(
                model=self.model, prompt=prompt, max_tokens=50, num_generations=1, stop_sequences=["\n"]
            )
            keywords = request.generations[0].text.strip()
            keywords = [keyword.strip() for keyword in keywords.split(",")]
            all_keywords.append(keywords)

        return all_keywords
