from types import SimpleNamespace

from src.rag.citations import format_citations


def test_format_citations_deduplicates():
    docs = [
        SimpleNamespace(
            metadata={"document_name": "Doc A", "page_number": 1},
            page_content="Fidelity expects a cyclical recession."
        ),
        SimpleNamespace(
            metadata={"document_name": "Doc A", "page_number": 1},
            page_content="Fidelity expects a cyclical recession."
        ),
        SimpleNamespace(
            metadata={"document_name": "Doc B", "page_number": 2},
            page_content="Bonds offer attractive yields."
        ),
    ]

    answer = "Fidelity expects a cyclical recession."

    citations = format_citations(docs, answer)

    assert citations == ["[Doc A, p. 1]"]