"""Build Silver layer: cleaned, chunked, metadata-rich records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.paths import ensure_directories


def clean_text(text: str) -> str:
    """Apply light text normalization for PDF-extracted content.

    Args:
        text: Raw extracted page text.

    Returns:
        Cleaned text.
    """
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_splitter(chunking_config: Dict[str, Any]) -> RecursiveCharacterTextSplitter:
    """Create recursive text splitter from config.

    Args:
        chunking_config: Chunking settings from config.

    Returns:
        Configured LangChain text splitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunking_config["chunk_size"],
        chunk_overlap=chunking_config["chunk_overlap"],
        separators=chunking_config["separators"],
        length_function=len,
    )


def chunk_bronze_document(
    bronze_record: Dict[str, Any],
    splitter: RecursiveCharacterTextSplitter,
) -> List[Dict[str, Any]]:
    """Convert a Bronze document record into Silver chunk records.

    Args:
        bronze_record: Bronze document JSON record.
        splitter: Configured recursive text splitter.

    Returns:
        List of chunk dictionaries.
    """
    silver_chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    for page in bronze_record["pages"]:
        raw_text = page.get("text", "")
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            continue

        page_chunks = splitter.split_text(cleaned_text)

        for local_chunk_index, chunk_text in enumerate(page_chunks):
            chunk_id = f"{bronze_record['doc_id']}_p{page['page_number']}_c{local_chunk_index}"

            silver_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": bronze_record["doc_id"],
                    "document_name": bronze_record["document_name"],
                    "source_file": bronze_record["source_file"],
                    "publisher": bronze_record["publisher"],
                    "year": bronze_record["year"],
                    "doc_type": bronze_record["doc_type"],
                    "page_number": page["page_number"],
                    "chunk_index": chunk_counter,
                    "page_chunk_index": local_chunk_index,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "section_title": None,
                }
            )
            chunk_counter += 1

    return silver_chunks


def build_silver() -> None:
    """Build Silver chunk artifacts from Bronze JSON documents."""
    config = load_config()
    ensure_directories(config["paths"])

    bronze_dir = Path(config["paths"]["bronze_dir"])
    silver_dir = Path(config["paths"]["silver_dir"])

    splitter = build_splitter(config["chunking"])

    bronze_files = sorted(bronze_dir.glob("*.json"))
    if not bronze_files:
        raise FileNotFoundError(f"No Bronze JSON files found in {bronze_dir}")

    for bronze_path in tqdm(bronze_files, desc="Building Silver layer"):
        with bronze_path.open("r", encoding="utf-8") as f:
            bronze_record = json.load(f)

        silver_chunks = chunk_bronze_document(bronze_record, splitter)

        output_path = silver_dir / f"{bronze_record['doc_id']}_chunks.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(silver_chunks, f, ensure_ascii=False, indent=2)

        print(
            f"Saved Silver file: {output_path.name} | "
            f"chunks={len(silver_chunks)} | "
            f"document={bronze_record['document_name']}"
        )


if __name__ == "__main__":
    build_silver()