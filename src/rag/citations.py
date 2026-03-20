"""Citation formatting utilities."""

from __future__ import annotations

from typing import List


def format_citations(retrieved_docs: List, answer: str) -> List[str]:
    """Return only citations that appear relevant to the answer."""
    citations = []

    for doc in retrieved_docs:
        document_name = doc.metadata.get("document_name", "Unknown Document")
        page_number = doc.metadata.get("page_number", "?")
        citation = f"[{document_name}, p. {page_number}]"

        # simple heuristic: include if page number appears in answer OR chunk text overlaps
        if str(page_number) in answer or doc.page_content[:100] in answer:
            citations.append(citation)

    return list(dict.fromkeys(citations))