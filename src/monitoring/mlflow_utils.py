"""MLflow logging utilities for AlphaQuery."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import mlflow

from src.utils.config import load_config


def setup_mlflow() -> bool:
    """Initialize MLflow experiment if enabled."""
    config = load_config()
    mlflow_config = config.get("mlflow", {})

    if not mlflow_config.get("enabled", False):
        return False

    experiment_name = mlflow_config.get("experiment_name", "alphaquery-rag")
    mlflow.set_experiment(experiment_name)
    return True


def _serialize_filters(filters: Dict[str, Any]) -> str:
    """Serialize filters for logging."""
    return json.dumps(filters, ensure_ascii=False, sort_keys=True)


def _extract_retrieval_metadata(retrieved_docs: List) -> Dict[str, Any]:
    """Extract retrieval metadata for MLflow logging."""
    document_names = []
    page_numbers = []

    for doc in retrieved_docs:
        document_names.append(doc.metadata.get("document_name"))
        page_numbers.append(doc.metadata.get("page_number"))

    return {
        "retrieved_document_names": json.dumps(document_names, ensure_ascii=False),
        "retrieved_page_numbers": json.dumps(page_numbers, ensure_ascii=False),
        "retrieved_chunk_count": len(retrieved_docs),
    }


def log_query_run(
    question: str,
    result: Dict[str, Any],
    filters: Optional[Dict[str, Any]] = None,
) -> None:
    """Log one query run to MLflow."""
    if not setup_mlflow():
        return

    filters = filters or {}
    retrieved_docs = result.get("retrieved_docs", [])
    llm_metadata = result.get("llm_metadata") or {}
    retrieval_meta = _extract_retrieval_metadata(retrieved_docs)

    with mlflow.start_run(nested=True):
        mlflow.log_param("question", question)
        mlflow.log_param("filters", _serialize_filters(filters))
        mlflow.log_param("llm_provider", llm_metadata.get("provider"))
        mlflow.log_param("model_name", llm_metadata.get("model_name"))

        mlflow.log_metric("latency_seconds", result.get("latency_seconds", 0.0))
        mlflow.log_metric("retrieved_chunk_count", retrieval_meta["retrieved_chunk_count"])

        mlflow.log_text(result.get("answer", ""), "artifacts/answer.txt")
        mlflow.log_text(
            json.dumps(result.get("citations", []), indent=2, ensure_ascii=False),
            "artifacts/citations.json",
        )
        mlflow.log_text(
            retrieval_meta["retrieved_document_names"],
            "artifacts/retrieved_document_names.json",
        )
        mlflow.log_text(
            retrieval_meta["retrieved_page_numbers"],
            "artifacts/retrieved_page_numbers.json",
        )


def log_evaluation_summary(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    """Log evaluation summary and detailed results to MLflow."""
    if not setup_mlflow():
        return

    with mlflow.start_run(run_name="evaluation_summary", nested=True):
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
            else:
                mlflow.log_param(key, str(value))

        mlflow.log_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            "artifacts/evaluation_results.json",
        )