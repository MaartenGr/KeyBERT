import time
from tqdm import tqdm
from litellm import completion
from typing import Mapping, Any, List
from keybert.llm._base import BaseLLM
from keybert.llm._utils import process_candidate_keywords


DEFAULT_PROMPT = """
I have the following document:
[DOCUMENT]

Based on the information above, extract the keywords that best describe the topic of the text.
Use the following format separated by commas:
<keywords>
"""


class LiteLLM(BaseLLM):
    r"""Extract keywords using LiteLLM to call any LLM API using OpenAI format
    such as Anthropic, Huggingface, Cohere, TogetherAI, Azure, OpenAI, etc.

    NOTE: The resulting keywords are expected to be separated by commas so
    any changes to the prompt will have to make sure that the resulting
    keywords are comma-separated.

    Arguments:
        model: Model to use within LiteLLM, defaults to OpenAI's `"gpt-3.5-turbo"`.
        generator_kwargs: Kwargs passed to `litellm.completion`
                          for fine-tuning the output.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[DOCUMENT]"` in the prompt
                to decide where the document needs to be inserted
        system_prompt: The message that sets the behavior of the assistant.
                       It's typically used to provide high-level instructions
                       for the conversation.
        delay_in_seconds: The delay in seconds between consecutive prompts
                          in order to prevent RateLimitErrors.
        verbose: Set this to True if you want to see a progress bar for the
                 keyword extraction.

    Usage:

    Let's use OpenAI as an example:

    ```python
    import os
    from keybert.llm import LiteLLM
    from keybert import KeyLLM

    # Select LLM
    os.environ["OPENAI_API_KEY"] = "sk-..."
    llm = LiteLLM("gpt-3.5-turbo")

    # Load it in KeyLLM
    kw_model = KeyLLM(llm)

    # Extract keywords
    document = "The website mentions that it only takes a couple of days to deliver but I still have not received mine."
    keywords = kw_model.extract_keywords(document)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following document: [DOCUMENT] \nThis document contains the following keywords separated by commas: '"
    llm = LiteLLM("gpt-3.5-turbo", prompt=prompt)
    ```
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        prompt: str = None,
        system_prompt: str = "You are a helpful assistant.",
        generator_kwargs: Mapping[str, Any] = {},
        delay_in_seconds: float = None,
        verbose: bool = False,
    ):
        self.model = model

        if prompt is None:
            self.prompt = DEFAULT_PROMPT
        else:
            self.prompt = prompt

        self.system_prompt = system_prompt
        self.default_prompt_ = DEFAULT_PROMPT
        self.delay_in_seconds = delay_in_seconds
        self.verbose = verbose

        self.generator_kwargs = generator_kwargs
        if self.generator_kwargs.get("model"):
            self.model = generator_kwargs.get("model")
        if self.generator_kwargs.get("prompt"):
            del self.generator_kwargs["prompt"]

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

            # Use a chat model
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
            kwargs = {"model": self.model, "messages": messages, **self.generator_kwargs}

            response = completion(**kwargs)
            keywords = response["choices"][0]["message"]["content"].strip()
            keywords = [keyword.strip() for keyword in keywords.split(",")]
            all_keywords.append(keywords)

        return all_keywords
