from src.ingestion.extract_pdf import infer_publisher, infer_year


def test_infer_publisher():
    assert infer_publisher("BlackRock 2024 Global Outlook.pdf") == "BlackRock"
    assert infer_publisher("Vanguard Outlook 2024.pdf") == "Vanguard"


def test_infer_year():
    assert infer_year("BlackRock 2024 Global Outlook.pdf") == 2024
    assert infer_year("Some Document.pdf") is None