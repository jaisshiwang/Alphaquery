"""Gold set evaluator for AlphaQuery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.evaluation.metrics import (
    answer_passes,
    document_hit,
    page_hit,
    token_overlap_score,
)
from src.rag.chain import answer_question
from src.monitoring.mlflow_utils import log_evaluation_summary
from src.utils.config import load_config


def load_gold_set(path: str) -> List[Dict[str, Any]]:
    """Load evaluation gold set JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one gold-set example."""
    filters = example.get("filters", {})

    result = answer_question(
        question=example["question"],
        publisher=filters.get("publisher"),
        year=filters.get("year"),
        document_name=filters.get("document_name"),
    )

    predicted_answer = result["answer"]
    retrieved_docs = result["retrieved_docs"]

    overlap = token_overlap_score(example["expected_answer"], predicted_answer)
    answer_ok = answer_passes(example["expected_answer"], predicted_answer)
    doc_ok = document_hit(example["expected_document"], retrieved_docs)
    page_ok = page_hit(example["expected_pages"], retrieved_docs)

    return {
        "id": example["id"],
        "question": example["question"],
        "expected_document": example["expected_document"],
        "expected_pages": example["expected_pages"],
        "predicted_answer": predicted_answer,
        "citations": result["citations"],
        "latency_seconds": result["latency_seconds"],
        "retrieved_chunk_count": len(retrieved_docs),
        "document_hit": doc_ok,
        "page_hit": page_ok,
        "token_overlap_score": round(overlap, 3),
        "answer_pass": answer_ok,
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize evaluation results."""
    total = len(results)

    return {
        "total_questions": total,
        "document_hit_rate": round(sum(r["document_hit"] for r in results) / total, 3),
        "page_hit_rate": round(sum(r["page_hit"] for r in results) / total, 3),
        "answer_pass_rate": round(sum(r["answer_pass"] for r in results) / total, 3),
        "avg_token_overlap": round(
            sum(r["token_overlap_score"] for r in results) / total, 3
        ),
        "avg_latency_seconds": round(
            sum(r["latency_seconds"] for r in results) / total, 3
        ),
    }


def main() -> None:
    """Run evaluation on the configured gold set."""
    config = load_config()
    gold_set_path = Path(config["paths"]["evaluation_dir"]) / "gold_set.json"

    gold_set = load_gold_set(str(gold_set_path))
    results = [evaluate_example(example) for example in gold_set]
    summary = summarize_results(results)

    log_evaluation_summary(summary, results)
    
    df = pd.DataFrame(results)

    print("\n=== Evaluation Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n=== Per-Question Results ===")
    print(
        df[
            [
                "id",
                "document_hit",
                "page_hit",
                "token_overlap_score",
                "answer_pass",
                "latency_seconds",
            ]
        ].to_string(index=False)
    )

    output_path = Path(config["paths"]["evaluation_dir"]) / "evaluation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()