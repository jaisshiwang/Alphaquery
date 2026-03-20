"""Build Bronze layer from raw PDF documents."""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from src.ingestion.extract_pdf import extract_pdf_pages
from src.utils.config import load_config
from src.utils.paths import ensure_directories


def build_bronze() -> None:
    """Extract PDFs into Bronze page-level JSON artifacts."""
    config = load_config()
    ensure_directories(config["paths"])

    documents_dir = Path(config["paths"]["documents_dir"])
    bronze_dir = Path(config["paths"]["bronze_dir"])

    pdf_files = sorted(documents_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {documents_dir}")

    for pdf_path in tqdm(pdf_files, desc="Building Bronze layer"):
        document_record = extract_pdf_pages(str(pdf_path))
        output_path = bronze_dir / f"{document_record['doc_id']}.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(document_record, f, ensure_ascii=False, indent=2)

        non_empty_pages = sum(1 for p in document_record["pages"] if p["text"])
        print(
            f"Saved Bronze file: {output_path.name} | "
            f"pages={document_record['page_count']} | "
            f"non_empty_pages={non_empty_pages}"
        )


if __name__ == "__main__":
    build_bronze()