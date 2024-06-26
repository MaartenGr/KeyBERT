import time
import openai
from tqdm import tqdm
from typing import Mapping, Any, List
from keybert.llm._base import BaseLLM
from keybert.llm._utils import retry_with_exponential_backoff, process_candidate_keywords


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

DEFAULT_CHAT_PROMPT = """
I have the following document:
[DOCUMENT]

Based on the information above, extract the keywords that best describe the topic of the text.
Use the following format separated by commas:
<keywords>
"""


class OpenAI(BaseLLM):
    r"""Using the OpenAI API to extract keywords.

    The default method is `openai.Completion` if `chat=False`.
    The prompts will also need to follow a completion task. If you
    are looking for a more interactive chats, use `chat=True`
    with `model=gpt-3.5-turbo`.

    For an overview see:
    https://platform.openai.com/docs/models

    NOTE: The resulting keywords are expected to be separated by commas so
    any changes to the prompt will have to make sure that the resulting
    keywords are comma-separated.

    Arguments:
        client: A `openai.OpenAI` client
        model: Model to use within OpenAI, defaults to `"text-ada-001"`.
               NOTE: If a `gpt-3.5-turbo` model is used, make sure to set
               `chat` to True.
        generator_kwargs: Kwargs passed to `openai.Completion.create`
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
        exponential_backoff: Retry requests with a random exponential backoff.
                             A short sleep is used when a rate limit error is hit,
                             then the requests is retried. Increase the sleep length
                             if errors are hit until 10 unsuccesfull requests.
                             If True, overrides `delay_in_seconds`.
        chat: Set this to True if a chat model is used. Generally, this GPT 3.5 or higher
              See: https://platform.openai.com/docs/models/gpt-3-5
        verbose: Set this to True if you want to see a progress bar for the
                 keyword extraction.

    Usage:

    To use this, you will need to install the openai package first:

    `pip install openai`

    Then, get yourself an API key and use OpenAI's API as follows:

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

    You can also use a custom prompt:

    ```python
    prompt = "I have the following document: [DOCUMENT] \nThis document contains the following keywords separated by commas: '"
    llm = OpenAI(client, prompt=prompt, delay_in_seconds=5)
    ```

    If you want to use OpenAI's ChatGPT model:

    ```python
    llm = OpenAI(client, model="gpt-3.5-turbo", delay_in_seconds=10, chat=True)
    ```
    """

    def __init__(
        self,
        client,
        model: str = "gpt-3.5-turbo-instruct",
        prompt: str = None,
        system_prompt: str = "You are a helpful assistant.",
        generator_kwargs: Mapping[str, Any] = {},
        delay_in_seconds: float = None,
        exponential_backoff: bool = False,
        chat: bool = False,
        verbose: bool = False,
    ):
        self.client = client
        self.model = model

        if prompt is None:
            self.prompt = DEFAULT_CHAT_PROMPT if chat else DEFAULT_PROMPT
        else:
            self.prompt = prompt

        self.system_prompt = system_prompt
        self.default_prompt_ = DEFAULT_CHAT_PROMPT if chat else DEFAULT_PROMPT
        self.delay_in_seconds = delay_in_seconds
        self.exponential_backoff = exponential_backoff
        self.chat = chat
        self.verbose = verbose

        self.generator_kwargs = generator_kwargs
        if self.generator_kwargs.get("model"):
            self.model = generator_kwargs.get("model")
        if self.generator_kwargs.get("prompt"):
            del self.generator_kwargs["prompt"]
        if not self.generator_kwargs.get("stop") and not chat:
            self.generator_kwargs["stop"] = "\n"

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
            if self.chat:
                messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
                kwargs = {"model": self.model, "messages": messages, **self.generator_kwargs}
                if self.exponential_backoff:
                    response = chat_completions_with_backoff(self.client, **kwargs)
                else:
                    response = self.client.chat.completions.create(**kwargs)
                keywords = response.choices[0].message.content.strip()

            # Use a non-chat model
            else:
                if self.exponential_backoff:
                    response = completions_with_backoff(
                        self.client, model=self.model, prompt=prompt, **self.generator_kwargs
                    )
                else:
                    response = self.client.completions.create(model=self.model, prompt=prompt, **self.generator_kwargs)
                keywords = response.choices[0].text.strip()
            keywords = [keyword.strip() for keyword in keywords.split(",")]
            all_keywords.append(keywords)

        return all_keywords


def completions_with_backoff(client, **kwargs):
    return retry_with_exponential_backoff(
        client.completions.create,
        errors=(openai.RateLimitError,),
    )(**kwargs)


def chat_completions_with_backoff(client, **kwargs):
    return retry_with_exponential_backoff(
        client.chat.completions.create,
        errors=(openai.RateLimitError,),
    )(**kwargs)
