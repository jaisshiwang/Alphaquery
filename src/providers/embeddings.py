"""Embedding provider factory."""

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDINGS = None  # Global variable to hold the loaded embedding model instance

def get_embeddings(embedding_config: dict):
    """Create embedding model from config."""
    global EMBEDDINGS

    if EMBEDDINGS is not None:
        return EMBEDDINGS

    provider = embedding_config["provider"]

    if provider == "sentence_transformers":
        EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=embedding_config["model_name"]
        )
        return EMBEDDINGS

    raise ValueError(f"Unsupported embedding provider: {provider}")