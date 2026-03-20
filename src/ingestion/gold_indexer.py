"""Build Gold layer: vector index from Silver chunk artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from tqdm import tqdm

from src.providers.embeddings import get_embeddings
from src.utils.config import load_config
from src.utils.paths import ensure_directories


def load_silver_chunks(silver_dir: Path) -> List[dict]:
    """Load all Silver chunk records from disk.

    Args:
        silver_dir: Directory containing Silver JSON files.

    Returns:
        List of chunk dictionaries.
    """
    all_chunks: List[dict] = []

    silver_files = sorted(silver_dir.glob("*_chunks.json"))
    if not silver_files:
        raise FileNotFoundError(f"No Silver chunk files found in {silver_dir}")

    for silver_path in tqdm(silver_files, desc="Loading Silver chunks"):
        with silver_path.open("r", encoding="utf-8") as f:
            chunk_records = json.load(f)
        all_chunks.extend(chunk_records)

    return all_chunks


def convert_chunks_to_documents(chunk_records: List[dict]) -> List[Document]:
    """Convert chunk dictionaries to LangChain Document objects.

    Args:
        chunk_records: Silver chunk records.

    Returns:
        List of LangChain Documents.
    """
    documents: List[Document] = []

    for chunk in chunk_records:
        metadata = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "document_name": chunk["document_name"],
            "source_file": chunk["source_file"],
            "publisher": chunk["publisher"],
            "year": chunk["year"],
            "doc_type": chunk["doc_type"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "page_chunk_index": chunk["page_chunk_index"],
            "section_title": chunk.get("section_title"),
        }

        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=metadata,
            )
        )

    return documents


def build_gold_index() -> None:
    """Create Chroma vector index from Silver chunks."""
    config = load_config()
    ensure_directories(config["paths"])

    silver_dir = Path(config["paths"]["silver_dir"])
    chroma_dir = config["paths"]["chroma_dir"]

    chunk_records = load_silver_chunks(silver_dir)
    documents = convert_chunks_to_documents(chunk_records)
    embeddings = get_embeddings(config["embeddings"])

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name="alphaquery_financial_docs",
    )

    vectorstore.persist()

    print(
        f"Gold index created successfully | documents={len(documents)} | "
        f"persist_directory={chroma_dir}"
    )


if __name__ == "__main__":
    build_gold_index()