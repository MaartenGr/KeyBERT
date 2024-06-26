import pytest
from keybert import KeyBERT
from keybert.backend import SentenceTransformerBackend
import sentence_transformers

from sklearn.feature_extraction.text import CountVectorizer
from .utils import get_test_data


doc_one, doc_two = get_test_data()


@pytest.mark.parametrize("keyphrase_length", [(1, i + 1) for i in range(5)])
@pytest.mark.parametrize("vectorizer", [None, CountVectorizer(ngram_range=(1, 1), stop_words="english")])
def test_single_doc_sentence_transformer_backend(keyphrase_length, vectorizer):
    """Test whether the keywords are correctly extracted"""
    top_n = 5

    model_name = "paraphrase-MiniLM-L6-v2"
    st_model = sentence_transformers.SentenceTransformer(model_name, device="cpu")

    kb_model = KeyBERT(model=SentenceTransformerBackend(st_model, batch_size=128))

    keywords = kb_model.extract_keywords(
        doc_one,
        keyphrase_ngram_range=keyphrase_length,
        min_df=1,
        top_n=top_n,
        vectorizer=vectorizer,
    )

    assert model_name in kb_model.model.embedding_model.tokenizer.name_or_path
    assert isinstance(keywords, list)
    assert isinstance(keywords[0], tuple)
    assert isinstance(keywords[0][0], str)
    assert isinstance(keywords[0][1], float)
    assert len(keywords) == top_n
    for keyword in keywords:
        assert len(keyword[0].split(" ")) <= keyphrase_length[1]
