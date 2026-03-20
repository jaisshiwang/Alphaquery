from types import SimpleNamespace

from src.rag.citations import format_citations


def test_format_citations_deduplicates():
    docs = [
        SimpleNamespace(metadata={"document_name": "Doc A", "page_number": 1}),
        SimpleNamespace(metadata={"document_name": "Doc A", "page_number": 1}),
        SimpleNamespace(metadata={"document_name": "Doc B", "page_number": 2}),
    ]

    citations = format_citations(docs)

    assert citations == ["[Doc A, p. 1]", "[Doc B, p. 2]"]