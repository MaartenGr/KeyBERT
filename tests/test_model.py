import pytest
from keybert import KeyBERT
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from .utils import get_test_data


doc_one, doc_two = get_test_data()
docs = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))["data"]
model = KeyBERT(model="all-MiniLM-L6-v2")


@pytest.mark.parametrize("keyphrase_length", [(1, i + 1) for i in range(5)])
@pytest.mark.parametrize("vectorizer", [None, CountVectorizer(ngram_range=(1, 1), stop_words="english")])
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
@pytest.mark.parametrize("vectorizer", [None, CountVectorizer(ngram_range=(1, 1), stop_words="english")])
@pytest.mark.parametrize("candidates", [None, ["praise"]])
@pytest.mark.parametrize("seed_keywords", [None, ["time", "night", "day", "moment"]])
def test_extract_keywords_single_doc(keyphrase_length, mmr, maxsum, vectorizer, candidates, seed_keywords):
    """Test extraction of protected single document method"""
    top_n = 5
    keywords = model.extract_keywords(
        doc_one,
        top_n=top_n,
        candidates=candidates,
        keyphrase_ngram_range=keyphrase_length,
        seed_keywords=seed_keywords,
        use_mmr=mmr,
        use_maxsum=maxsum,
        diversity=0.5,
        vectorizer=vectorizer,
    )
    assert isinstance(keywords, list)
    if not candidates:
        assert isinstance(keywords[0][0], str)
        assert isinstance(keywords[0][1], float)
        assert len(keywords) == top_n
    for keyword in keywords:
        assert len(keyword[0].split(" ")) <= keyphrase_length[1]

    if candidates and keyphrase_length[1] == 1 and not vectorizer and not maxsum:
        assert keywords[0][0] == candidates[0]


@pytest.mark.parametrize("keyphrase_length", [(1, i + 1) for i in range(5)])
@pytest.mark.parametrize("candidates", [None, ["praise"]])
def test_extract_keywords_multiple_docs(keyphrase_length, candidates):
    """Test extraction of protected multiple document method"""
    top_n = 5
    keywords_list = model.extract_keywords(
        [doc_one, doc_two], top_n=top_n, keyphrase_ngram_range=keyphrase_length, candidates=candidates
    )
    assert isinstance(keywords_list, list)
    assert isinstance(keywords_list[0], list)
    assert len(keywords_list) == 2

    if not candidates:
        for keywords in keywords_list:
            assert len(keywords) == top_n

            for keyword in keywords:
                assert len(keyword[0].split(" ")) <= keyphrase_length[1]

    if candidates and keyphrase_length[1] == 1:
        assert keywords_list[0][0][0] == candidates[0]
        assert len(keywords_list[1]) == 0


def test_guided():
    """Test whether the keywords are correctly extracted"""

    # single doc + a keywords list
    top_n = 5
    seed_keywords = ["time", "night", "day", "moment"]
    keywords = model.extract_keywords(doc_one, min_df=1, top_n=top_n, seed_keywords=seed_keywords)
    assert isinstance(keywords, list)
    assert isinstance(keywords[0], tuple)
    assert isinstance(keywords[0][0], str)
    assert isinstance(keywords[0][1], float)
    assert len(keywords) == top_n

    # a bacth of docs sharing one single list of seed keywords
    top_n = 5
    list_of_docs = [doc_one, doc_two]
    list_of_seed_keywords = ["time", "night", "day", "moment"]
    keywords = model.extract_keywords(list_of_docs, min_df=1, top_n=top_n, seed_keywords=list_of_seed_keywords)
    print(keywords)

    assert isinstance(keywords, list)
    assert isinstance(keywords[0], list)
    assert isinstance(keywords[0][0], tuple)
    assert isinstance(keywords[0][0][0], str)
    assert isinstance(keywords[0][0][1], float)
    assert len(keywords[0]) == top_n

    # a bacth of docs, each of which has its own seed keywords
    top_n = 5
    list_of_docs = [doc_one, doc_two]
    list_of_seed_keywords = [["time", "night", "day", "moment"], ["hockey", "games", "afternoon", "tv"]]
    keywords = model.extract_keywords(list_of_docs, min_df=1, top_n=top_n, seed_keywords=list_of_seed_keywords)
    print(keywords)

    assert isinstance(keywords, list)
    assert isinstance(keywords[0], list)
    assert isinstance(keywords[0][0], tuple)
    assert isinstance(keywords[0][0][0], str)
    assert isinstance(keywords[0][0][1], float)
    assert len(keywords[0]) == top_n


def test_empty_doc():
    """Test empty document"""
    doc = ""
    result = model.extract_keywords(doc)

    assert result == []


def test_extract_embeddings():
    """Test extracting embeddings and testing out different parameters"""
    n_docs = 50
    doc_embeddings, word_embeddings = model.extract_embeddings(docs[:n_docs])
    keywords_fast = model.extract_keywords(
        docs[:n_docs], doc_embeddings=doc_embeddings, word_embeddings=word_embeddings
    )
    keywords_slow = model.extract_keywords(docs[:n_docs])

    assert doc_embeddings.shape[1] == word_embeddings.shape[1]
    assert doc_embeddings.shape[0] == n_docs
    assert keywords_fast == keywords_slow

    # When we use `min_df=3` to extract the keywords, this should give an error since
    # this value was not used when extracting the embeddings and should be the same.
    with pytest.raises(ValueError):
        _ = model.extract_keywords(
            docs[:n_docs], doc_embeddings=doc_embeddings, word_embeddings=word_embeddings, min_df=3
        )
