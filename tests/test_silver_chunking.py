from src.ingestion.silver_chunker import clean_text


def test_clean_text():
    raw = "Hello   world\n\n\nThis is\xa0a test."
    cleaned = clean_text(raw)
    assert "  " not in cleaned
    assert "\xa0" not in cleaned
    assert cleaned.startswith("Hello world")