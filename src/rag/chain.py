"""End-to-end RAG query pipeline."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.monitoring.mlflow_utils import log_query_run
from src.rag.citations import format_citations
from src.rag.generator import generate_answer
from src.rag.retriever import retrieve_chunks


def answer_question(
    question: str,
    top_k: Optional[int] = None,
    publisher: Optional[str] = None,
    year: Optional[int] = None,
    document_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run retrieval + grounded generation for a user question."""
    start_time = time.time()

    filters = {
        "publisher": publisher,
        "year": year,
        "document_name": document_name,
    }

    retrieved_docs = retrieve_chunks(
        query=question,
        top_k=top_k,
        publisher=publisher,
        year=year,
        document_name=document_name,
    )

    if not retrieved_docs:
        result = {
            "answer": "No relevant evidence was retrieved for this question.",
            "citations": [],
            "retrieved_docs": [],
            "latency_seconds": round(time.time() - start_time, 3),
            "llm_metadata": None,
        }
        log_query_run(question=question, result=result, filters=filters)
        return result

    answer, llm_metadata = generate_answer(question, retrieved_docs)
    citations = format_citations(retrieved_docs, answer)

    result = {
        "answer": answer,
        "citations": citations,
        "retrieved_docs": retrieved_docs,
        "latency_seconds": round(time.time() - start_time, 3),
        "llm_metadata": llm_metadata,
    }

    log_query_run(question=question, result=result, filters=filters)
    return result