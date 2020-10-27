import pytest
from .utils import get_test_data

doc_one, doc_two = get_test_data()


@pytest.mark.parametrize("keyphrase_length", [i+1 for i in range(5)])
def test_single_doc(keyphrase_length, base_keybert):
    """ Test whether the keywords are correctly extracted """
    top_n = 5
    keywords = base_keybert.extract_keywords(doc_one, keyphrase_length=keyphrase_length, min_df=1, top_n=top_n)
    assert isinstance(keywords, list)
    assert isinstance(keywords[0], str)
    assert len(keywords) == top_n
    for keyword in keywords:
        assert len(keyword.split(" ")) == keyphrase_length


@pytest.mark.parametrize("keyphrase_length, mmr", [(i+1, truth) for i in range(5) for truth in [True, False]])
def test_extract_keywords_single_doc(keyphrase_length, mmr, base_keybert):
    """ Test extraction of protected single document method """
    top_n = 5
    keywords = base_keybert._extract_keywords_single_doc(doc_one,
                                                         top_n=top_n,
                                                         keyphrase_length=keyphrase_length,
                                                         use_mmr=mmr,
                                                         diversity=0.5)
    assert isinstance(keywords, list)
    assert isinstance(keywords[0], str)
    assert len(keywords) == top_n
    for keyword in keywords:
        assert len(keyword.split(" ")) == keyphrase_length


@pytest.mark.parametrize("keyphrase_length", [i+1 for i in range(5)])
def test_extract_keywords_multiple_docs(keyphrase_length, base_keybert):
    """ Test extractino of protected multiple document method"""
    top_n = 5
    keywords_list = base_keybert._extract_keywords_multiple_docs([doc_one, doc_two],
                                                                 top_n=top_n,
                                                                 keyphrase_length=keyphrase_length)
    assert isinstance(keywords_list, list)
    assert isinstance(keywords_list[0], list)
    assert len(keywords_list) == 2

    for keywords in keywords_list:
        assert len(keywords) == top_n

        for keyword in keywords:
            assert len(keyword.split(" ")) == keyphrase_length


def test_error(base_keybert):
    """ Empty doc should raise a ValueError """
    with pytest.raises(AttributeError):
        doc = []
        base_keybert._extract_keywords_single_doc(doc)


def test_empty_doc(base_keybert):
    """ Test empty document """
    doc = ""
    result = base_keybert._extract_keywords_single_doc(doc)

    assert result == []
