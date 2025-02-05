from ._base import BaseEmbedder


def select_backend(embedding_model) -> BaseEmbedder:
    """Select an embedding model based on language or a specific sentence transformer models.
    When selecting a language, we choose `all-MiniLM-L6-v2` for English and
    `paraphrase-multilingual-MiniLM-L12-v2` for all other languages as it support 100+ languages.

    Returns:
        model: Either a Sentence-Transformer or Flair model
    """
    # keybert language backend
    if isinstance(embedding_model, BaseEmbedder):
        return embedding_model

    # Model2Vec embeddings
    if "model2vec" in str(type(embedding_model)):
        from keybert.backend._model2vec import Model2VecBackend

        return Model2VecBackend(embedding_model)

    # Flair word embeddings
    if "flair" in str(type(embedding_model)):
        from keybert.backend._flair import FlairBackend

        return FlairBackend(embedding_model)

    # Spacy embeddings
    if "spacy" in str(type(embedding_model)):
        from keybert.backend._spacy import SpacyBackend

        return SpacyBackend(embedding_model)

    # Gensim embeddings
    if "gensim" in str(type(embedding_model)):
        from keybert.backend._gensim import GensimBackend

        return GensimBackend(embedding_model)

    # USE embeddings
    if "tensorflow" and "saved_model" in str(type(embedding_model)):
        from keybert.backend._use import USEBackend

        return USEBackend(embedding_model)

    # Sentence Transformer embeddings
    if "sentence_transformers" in str(type(embedding_model)):
        from ._sentencetransformers import SentenceTransformerBackend

        return SentenceTransformerBackend(embedding_model)

    # Create a Sentence Transformer model based on a string
    if isinstance(embedding_model, str):
        from ._sentencetransformers import SentenceTransformerBackend

        return SentenceTransformerBackend(embedding_model)

    # Hugging Face embeddings
    try:
        from transformers.pipelines import Pipeline

        if isinstance(embedding_model, Pipeline):
            from ._hftransformers import HFTransformerBackend

            return HFTransformerBackend(embedding_model)
    except ImportError:
        pass

    from ._sentencetransformers import SentenceTransformerBackend

    return SentenceTransformerBackend("paraphrase-multilingual-MiniLM-L12-v2")
