from importlib.metadata import version

from keybert._llm import KeyLLM
from keybert._model import KeyBERT

__version__ = version("keybert")

__all__ = [
    "KeyBERT",
    "KeyLLM",
]
