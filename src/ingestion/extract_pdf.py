"""PDF extraction utilities for Bronze layer."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF


def infer_publisher(file_name: str) -> str:
    """Infer publisher from filename."""
    name = file_name.lower()
    if "blackrock" in name:
        return "BlackRock"
    if "fidelity" in name:
        return "Fidelity"
    if "pimco" in name:
        return "PIMCO"
    if "vanguard" in name:
        return "Vanguard"
    return "Unknown"


def infer_year(file_name: str) -> int | None:
    """Infer year from filename."""
    for token in file_name.replace(".pdf", "").split():
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def extract_pdf_pages(pdf_path: str) -> Dict:
    """Extract page-level text and metadata from a PDF.

    Args:
        pdf_path: Path to the PDF.

    Returns:
        Dictionary containing document metadata and extracted pages.
    """
    path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    pages: List[Dict] = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        text = page.get_text("text") or ""
        pages.append(
            {
                "page_number": page_index + 1,
                "text": text.strip(),
                "char_count": len(text.strip()),
            }
        )

    return {
        "doc_id": path.stem.lower().replace(" ", "_"),
        "document_name": path.stem,
        "source_file": path.name,
        "source_path": str(path),
        "publisher": infer_publisher(path.name),
        "year": infer_year(path.name),
        "doc_type": "financial_outlook",
        "page_count": len(doc),
        "pages": pages,
    }