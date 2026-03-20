from src.rag.retriever import build_filter


def test_build_filter_empty():
    assert build_filter() == {}


def test_build_filter_partial():
    assert build_filter(publisher="BlackRock") == {"publisher": "BlackRock"}


def test_build_filter_full():
    assert build_filter(
        publisher="BlackRock",
        year=2024,
        document_name="BlackRock 2024 Global Outlook",
    ) == {
        "publisher": "BlackRock",
        "year": 2024,
        "document_name": "BlackRock 2024 Global Outlook",
    }