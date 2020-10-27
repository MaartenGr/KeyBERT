from keybert import KeyBERT
import pytest


@pytest.fixture(scope="module")
def base_keybert():
    model = KeyBERT(model = 'distilbert-base-nli-mean-tokens')
    return model
