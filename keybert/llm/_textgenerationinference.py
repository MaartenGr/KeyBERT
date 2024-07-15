from tqdm import tqdm
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from typing import Mapping, List, Any
from keybert.llm._base import BaseLLM
from keybert.llm._utils import process_candidate_keywords
import json

DEFAULT_PROMPT = """
    I have the following document:
    [DOCUMENT]

    With the following candidate keywords:
    [CANDIDATES]

    Based on the information above, improve the candidate keywords to best describe the topic of the document.

    Output in JSON format:
"""


class Keywords(BaseModel):
    keywords: List[str]


class TextGenerationInference(BaseLLM):
    """Tex.

    Arguments:
        client: InferenceClient from huggingface_hub.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        client_kwargs: Kwargs that you can pass to the client.text_generation
                         when it is called.
        json_schema: Pydantic BaseModel to be used as guidance for keywords.
                By default uses:
                class Keywords(BaseModel):
                    keywords: List[str]

    Usage:

    ```python
    from pydantic import BaseModel
    from huggingface_hub import InferenceClient
    from keybert.llm import TextGenerationInference
    from keybert import KeyLLM

    # Json Schema
    class Keywords(BaseModel):
        keywords: List[str]

    # Create your LLM
    generator = InferenceClient('url')
    llm = TextGenerationInference(generator, Keywords)

    # Load it in KeyLLM
    kw_model = KeyLLM(llm)

    # Extract keywords
    document = "The website mentions that it only takes a couple of days to deliver but I still have not received mine."
    keywords = kw_model.extract_keywords(document)
    ```

    You can use a custom prompt and decide where the document should
    be inserted with the `[DOCUMENT]` tag:

    ```python
    from keybert.llm import TextGenerationInference

    prompt = "I have the following documents '[DOCUMENT]'. Please give me the keywords that are present in this document and separate them with commas:"

    # Create your representation model
    from huggingface_hub import InferenceClient
    generator = InferenceClient('url')
    llm = TextGenerationInference(generator)
    ```
    """

    def __init__(self, client: InferenceClient, prompt: str = None, json_schema: BaseModel = Keywords):
        self.client = client
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT
        self.json_schema = json_schema

    def extract_keywords(
        self, documents: List[str], candidate_keywords: List[List[str]] = None, inference_kwargs: Mapping[str, Any] = {}
    ):
        """Extract topics.

        Arguments:
            documents: The documents to extract keywords from
            candidate_keywords: A list of candidate keywords that the LLM will fine-tune
                        For example, it will create a nicer representation of
                        the candidate keywords, remove redundant keywords, or
                        shorten them depending on the input prompt.
            inference_kwargs: kwargs for `InferenceClient.text_generation`. See: https://huggingface.co/docs/huggingface_hub/package_reference/inference_client

        Returns:
            all_keywords: All keywords for each document
        """
        all_keywords = []
        candidate_keywords = process_candidate_keywords(documents, candidate_keywords)

        for document, candidates in tqdm(zip(documents, candidate_keywords), disable=not self.verbose):
            prompt = self.prompt.replace("[DOCUMENT]", document)
            if candidates is not None:
                prompt = prompt.replace("[CANDIDATES]", ", ".join(candidates))

            # Extract result from generator and use that as label
            response = self.client.text_generation(
                prompt=prompt, grammar={"type": "json", "value": self.json_schema.schema()}, **inference_kwargs
            )
            all_keywords = json.loads(response)["keywords"]

        return all_keywords
