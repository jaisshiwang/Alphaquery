"""Evaluation metrics for AlphaQuery."""

from __future__ import annotations

import re
from typing import List


def normalize_text(text: str) -> str:
    """Normalize text for simple comparison."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def token_overlap_score(expected: str, predicted: str) -> float:
    """Compute simple token overlap score."""
    expected_tokens = set(normalize_text(expected).split())
    predicted_tokens = set(normalize_text(predicted).split())

    if not expected_tokens:
        return 0.0

    overlap = expected_tokens.intersection(predicted_tokens)
    return len(overlap) / len(expected_tokens)


def answer_passes(expected: str, predicted: str, threshold: float = 0.5) -> bool:
    """Decide whether an answer passes based on token overlap."""
    return token_overlap_score(expected, predicted) >= threshold


def document_hit(expected_document: str, retrieved_docs: List) -> bool:
    """Check whether expected document appears in retrieved results."""
    retrieved_doc_names = {
        doc.metadata.get("document_name", "") for doc in retrieved_docs
    }
    return expected_document in retrieved_doc_names


def page_hit(expected_pages: List[int], retrieved_docs: List) -> bool:
    """Check whether any expected page appears in retrieved results."""
    retrieved_pages = {doc.metadata.get("page_number") for doc in retrieved_docs}
    return any(page in retrieved_pages for page in expected_pages)