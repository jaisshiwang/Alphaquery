"""Retriever utilities for AlphaQuery."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma

from src.providers.embeddings import get_embeddings
from src.utils.config import load_config

VECTORSTORE = None  # Global variable to hold the loaded vector store instance

def load_vectorstore() -> Chroma:
    """Load persisted Chroma vector store."""
    global VECTORSTORE
    if VECTORSTORE is None:
        config = load_config()
        embeddings = get_embeddings(config["embeddings"])

        VECTORSTORE =  Chroma(
            persist_directory=config["paths"]["chroma_dir"],
            embedding_function=embeddings,
            collection_name="alphaquery_financial_docs",
        )
    return VECTORSTORE


def build_filter(
    publisher: Optional[str] = None,
    year: Optional[int] = None,
    document_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build Chroma metadata filter dictionary."""
    filter_dict: Dict[str, Any] = {}

    if publisher:
        filter_dict["publisher"] = publisher
    if year:
        filter_dict["year"] = year
    if document_name:
        filter_dict["document_name"] = document_name

    return filter_dict


def retrieve_chunks(
    query: str,
    top_k: Optional[int] = None,
    publisher: Optional[str] = None,
    year: Optional[int] = None,
    document_name: Optional[str] = None,
) -> List:
    """Retrieve relevant chunks with optional metadata filtering."""
    config = load_config()
    vectorstore = load_vectorstore()

    k = top_k or config["retrieval"]["top_k"]
    filter_dict = build_filter(
        publisher=publisher,
        year=year,
        document_name=document_name,
    )

    search_kwargs: Dict[str, Any] = {"k": k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    return vectorstore.similarity_search(query, **search_kwargs)