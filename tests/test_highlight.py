import pytest
from sklearn.feature_extraction.text import CountVectorizer

from keybert._highlight import _highlight_n_gram


@pytest.mark.parametrize(
    "doc, keyword, expected",
    [
        ("art smart people", "art", "[black on #FFFF00]art[/] smart people"),
        ("very very good results", "very", "[black on #FFFF00]very[/] very good results"),
    ],
)
def test_highlight_n_gram_keeps_words_after_keyword(doc, keyword, expected):
    vectorizer = CountVectorizer(ngram_range=(1, 2)).fit([doc])

    assert _highlight_n_gram(doc, [keyword], vectorizer) == expected
