from typing import Tuple, List

try:
    from rich.console import Console

    HAS_RICH = True
except ModuleNotFoundError:
    HAS_RICH = False

from sklearn.feature_extraction.text import CountVectorizer


def highlight_document(doc: str, keywords: List[Tuple[str, float]], vectorizer: CountVectorizer):
    """Highlight keywords in a document.

    Arguments:
        doc: The document for which to extract keywords/keyphrases.
        keywords: The top n keywords for a document with their respective distances
                  to the input document.
        vectorizer: The vectorizer used for tokenizing the document.

    Returns:
        highlighted_text: The document with additional tags to highlight keywords
                          according to the rich package.
    """
    if not HAS_RICH:
        raise ModuleNotFoundError(
            "The `rich` package is required for highlighting which you can install with `pip install rich`."
        )
    keywords_only = [keyword for keyword, _ in keywords]
    max_len = vectorizer.ngram_range[1]

    if max_len == 1:
        highlighted_text = _highlight_one_gram(doc, keywords_only, vectorizer)
    else:
        highlighted_text = _highlight_n_gram(doc, keywords_only, vectorizer)

    from rich.highlighter import RegexHighlighter

    class NullHighlighter(RegexHighlighter):
        """Basic highlighter."""

        base_style = ""
        highlights = [r""]

    console = Console(highlighter=NullHighlighter())
    console.print(highlighted_text)


def _highlight_one_gram(doc: str, keywords: List[str], vectorizer: CountVectorizer) -> str:
    """Highlight 1-gram keywords in a document.

    Arguments:
        doc: The document for which to extract keywords/keyphrases.
        keywords: The top n keywords for a document.
        vectorizer: The vectorizer used for tokenizing the document.

    Returns:
        highlighted_text: The document with additional tags to highlight keywords
                          according to the rich package.
    """
    tokenizer = vectorizer.build_tokenizer()
    tokens = tokenizer(doc)
    separator = "" if "zh" in str(tokenizer) else " "

    highlighted_text = separator.join(
        [f"[black on #FFFF00]{token}[/]" if token.lower() in keywords else f"{token}" for token in tokens]
    ).strip()
    return highlighted_text


def _highlight_n_gram(doc: str, keywords: List[str], vectorizer: CountVectorizer) -> str:
    """Highlight n-gram keywords in a document.

    Arguments:
        doc: The document for which to extract keywords/keyphrases.
        keywords: The top n keywords for a document.
        vectorizer: The vectorizer used for tokenizing the document.

    Returns:
        highlighted_text: The document with additional tags to highlight keywords
                          according to the rich package.
    """
    tokenizer = vectorizer.build_tokenizer()
    tokens = tokenizer(doc)
    max_len = vectorizer.ngram_range[1]
    separator = "" if "zh" in str(tokenizer) else " "

    n_gram_tokens = [
        [separator.join(tokens[i : i + max_len][0 : j + 1]) for j in range(max_len)] for i, _ in enumerate(tokens)
    ]
    highlighted_text = []
    skip = False

    for n_grams in n_gram_tokens:
        candidate = False

        if not skip:
            for index, n_gram in enumerate(n_grams):
                if n_gram.lower() in keywords:
                    candidate = f"[black on #FFFF00]{n_gram}[/]" + n_grams[-1].split(n_gram)[-1]
                    skip = index + 1

            if not candidate:
                candidate = n_grams[0]

            highlighted_text.append(candidate)

        else:
            skip = skip - 1
    highlighted_text = separator.join(highlighted_text)
    return highlighted_text
