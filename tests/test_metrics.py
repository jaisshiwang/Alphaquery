from src.evaluation.metrics import (
    answer_passes,
    document_hit,
    page_hit,
    token_overlap_score,
)


class DummyDoc:
    def __init__(self, document_name, page_number):
        self.metadata = {
            "document_name": document_name,
            "page_number": page_number,
        }


def test_token_overlap_score():
    expected = "Fidelity base case is a cyclical recession"
    predicted = "Fidelity says the base case is a cyclical recession"
    score = token_overlap_score(expected, predicted)
    assert score > 0.5


def test_answer_passes():
    assert answer_passes(
        "cyclical recession",
        "The answer is cyclical recession.",
        threshold=0.5,
    )


def test_document_hit():
    docs = [DummyDoc("Fidelity International Outlook 2024", 5)]
    assert document_hit("Fidelity International Outlook 2024", docs)


def test_page_hit():
    docs = [DummyDoc("Fidelity International Outlook 2024", 5)]
    assert page_hit([5, 6], docs)