PYTHON := python
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

init:
	$(PYTHON_VENV) -m src.main

run:
	PYTHONPATH=. $(VENV)/bin/streamlit run app/streamlit_app.py

test:
	$(PYTHON_VENV) -m pytest

ingest:
	$(PYTHON_VENV) -m src.ingestion.bronze_builder

silver:
	$(PYTHON_VENV) -m src.ingestion.silver_chunker

gold:
	$(PYTHON_VENV) -m src.ingestion.gold_indexer

query:
	$(PYTHON_VENV) -m src.rag.run_query

eval:
	$(PYTHON_VENV) -m src.evaluation.evaluator

mlflow-ui:
	$(VENV)/bin/mlflow ui --backend-store-uri mlruns

bootstrap: setup init ingest silver gold