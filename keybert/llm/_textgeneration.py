from tqdm import tqdm
from transformers import pipeline, set_seed
from transformers.pipelines.base import Pipeline
from typing import Mapping, List, Any, Union
from keybert.llm._base import BaseLLM
from keybert.llm._utils import process_candidate_keywords


DEFAULT_PROMPT = """
I have the following document:
* [DOCUMENT]

Please give me the keywords that are present in this document and separate them with commas:
"""


class TextGeneration(BaseLLM):
    """Text2Text or text generation with transformers.

    NOTE: The resulting keywords are expected to be separated by commas so
    any changes to the prompt will have to make sure that the resulting
    keywords are comma-separated.

    Arguments:
        model: A transformers pipeline that should be initialized as "text-generation"
               for gpt-like models or "text2text-generation" for T5-like models.
               For example, `pipeline('text-generation', model='gpt2')`. If a string
               is passed, "text-generation" will be selected by default.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        pipeline_kwargs: Kwargs that you can pass to the transformers.pipeline
                         when it is called.
        random_state: A random state to be passed to `transformers.set_seed`
        verbose: Set this to True if you want to see a progress bar for the
                 keyword extraction.

    Usage:

    To use a gpt-like model:

    ```python
    from keybert.llm import TextGeneration
    from keybert import KeyLLM

    # Create your LLM
    generator = pipeline('text-generation', model='gpt2')
    llm = TextGeneration(generator)

    # Load it in KeyLLM
    kw_model = KeyLLM(llm)

    # Extract keywords
    document = "The website mentions that it only takes a couple of days to deliver but I still have not received mine."
    keywords = kw_model.extract_keywords(document)
    ```

    You can use a custom prompt and decide where the document should
    be inserted with the `[DOCUMENT]` tag:

    ```python
    from keybert.llm import TextGeneration

    prompt = "I have the following documents '[DOCUMENT]'. Please give me the keywords that are present in this document and separate them with commas:"

    # Create your representation model
    generator = pipeline('text2text-generation', model='google/flan-t5-base')
    llm = TextGeneration(generator)
    ```
    """

    def __init__(
        self,
        model: Union[str, pipeline],
        prompt: str = None,
        pipeline_kwargs: Mapping[str, Any] = {},
        random_state: int = 42,
        verbose: bool = False,
    ):
        set_seed(random_state)
        if isinstance(model, str):
            self.model = pipeline("text-generation", model=model)
        elif isinstance(model, Pipeline):
            self.model = model
        else:
            raise ValueError(
                "Make sure that the HF model that you"
                "pass is either a string referring to a"
                "HF model or a `transformers.pipeline` object."
            )
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT
        self.pipeline_kwargs = pipeline_kwargs
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

            # Extract result from generator and use that as label
            keywords = self.model(prompt, **self.pipeline_kwargs)[0]["generated_text"].replace(prompt, "")
            keywords = [keyword.strip() for keyword in keywords.split(",")]
            all_keywords.append(keywords)

        return all_keywords
