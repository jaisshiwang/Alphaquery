from types import SimpleNamespace

from src.rag.prompts import build_context_block


def test_build_context_block():
    docs = [
        SimpleNamespace(
            metadata={"document_name": "Doc A", "page_number": 1},
            page_content="This is the first chunk.",
        ),
        SimpleNamespace(
            metadata={"document_name": "Doc B", "page_number": 2},
            page_content="This is the second chunk.",
        ),
    ]

    context = build_context_block(docs)

    assert "Doc A" in context
    assert "page 1" in context
    assert "This is the first chunk." in context
    assert "Doc B" in context