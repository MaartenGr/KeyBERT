from keybert import KeyBERT
import pytest


@pytest.fixture(scope="module")
def base_keybert():
    model = KeyBERT(model='paraphrase-MiniLM-L6-v2')
    return model
