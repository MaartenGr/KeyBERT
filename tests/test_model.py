import pytest
from .utils import get_test_data
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT

doc_one, doc_two = get_test_data()
model = KeyBERT(model="all-MiniLM-L6-v2")


@pytest.mark.parametrize("keyphrase_length", [(1, i + 1) for i in range(5)])
@pytest.mark.parametrize(
    "vectorizer", [None, CountVectorizer(ngram_range=(1, 1), stop_words="english")]
)
def test_single_doc(keyphrase_length, vectorizer):
    """Test whether the keywords are correctly extracted"""
    top_n = 5

    keywords = model.extract_keywords(
        doc_one,
        keyphrase_ngram_range=keyphrase_length,
        min_df=1,
        top_n=top_n,
        vectorizer=vectorizer,
    )

    assert isinstance(keywords, list)
    assert isinstance(keywords[0], tuple)
    assert isinstance(keywords[0][0], str)
    assert isinstance(keywords[0][1], float)
    assert len(keywords) == top_n
    for keyword in keywords:
        assert len(keyword[0].split(" ")) <= keyphrase_length[1]


@pytest.mark.parametrize(
    "keyphrase_length, mmr, maxsum",
    [((1, i + 1), truth, not truth) for i in range(4) for truth in [True, False]],
)
@pytest.mark.parametrize(
    "vectorizer", [None, CountVectorizer(ngram_range=(1, 1), stop_words="english")]
)
def test_extract_keywords_single_doc(keyphrase_length, mmr, maxsum, vectorizer):
    """Test extraction of protected single document method"""
    top_n = 5
    keywords = model._extract_keywords_single_doc(
        doc_one,
        top_n=top_n,
        keyphrase_ngram_range=keyphrase_length,
        use_mmr=mmr,
        use_maxsum=maxsum,
        diversity=0.5,
        vectorizer=vectorizer,
    )
    assert isinstance(keywords, list)
    assert isinstance(keywords[0][0], str)
    assert isinstance(keywords[0][1], float)
    assert len(keywords) == top_n
    for keyword in keywords:
        assert len(keyword[0].split(" ")) <= keyphrase_length[1]


@pytest.mark.parametrize("keyphrase_length", [(1, i + 1) for i in range(5)])
def test_extract_keywords_multiple_docs(keyphrase_length):
    """Test extractino of protected multiple document method"""
    top_n = 5
    keywords_list = model._extract_keywords_multiple_docs(
        [doc_one, doc_two], top_n=top_n, keyphrase_ngram_range=keyphrase_length
    )
    assert isinstance(keywords_list, list)
    assert isinstance(keywords_list[0], list)
    assert len(keywords_list) == 2

    for keywords in keywords_list:
        assert len(keywords) == top_n

        for keyword in keywords:
            assert len(keyword[0].split(" ")) <= keyphrase_length[1]


def test_guided():
    """Test whether the keywords are correctly extracted"""
    top_n = 5
    seed_keywords = ["time", "night", "day", "moment"]
    keywords = model.extract_keywords(
        doc_one, min_df=1, top_n=top_n, seed_keywords=seed_keywords
    )

    assert isinstance(keywords, list)
    assert isinstance(keywords[0], tuple)
    assert isinstance(keywords[0][0], str)
    assert isinstance(keywords[0][1], float)
    assert len(keywords) == top_n


def test_error():
    """Empty doc should raise a ValueError"""
    with pytest.raises(AttributeError):
        doc = []
        model._extract_keywords_single_doc(doc)


def test_empty_doc():
    """Test empty document"""
    doc = ""
    result = model._extract_keywords_single_doc(doc)

    assert result == []
