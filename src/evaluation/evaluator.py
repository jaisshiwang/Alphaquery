"""Gold set evaluator for AlphaQuery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.evaluation.metrics import (
    answer_passes,
    document_hit,
    page_hit,
    token_overlap_score,
)
from src.evaluation.ragas_evaluator import run_ragas_evaluation, save_ragas_results
from src.monitoring.mlflow_utils import log_evaluation_summary
from src.rag.chain import answer_question
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
        "expected_answer": example["expected_answer"],
        "expected_document": example["expected_document"],
        "expected_pages": example["expected_pages"],
        "predicted_answer": predicted_answer,
        "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
        "citations": result["citations"],
        "latency_seconds": result["latency_seconds"],
        "retrieved_chunk_count": len(retrieved_docs),
        "document_hit": doc_ok,
        "page_hit": page_ok,
        "token_overlap_score": round(overlap, 3),
        "answer_pass": answer_ok,
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize baseline evaluation results."""
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


def merge_ragas_per_question(
    df: pd.DataFrame,
    ragas_df: Any,
) -> Tuple[pd.DataFrame, List[str]]:
    """Merge per-question RAGAS metrics into the baseline dataframe."""
    if not isinstance(ragas_df, pd.DataFrame):
        try:
            ragas_df = pd.DataFrame(ragas_df)
        except Exception as exc:
            raise TypeError(
                f"ragas_df must be a pandas DataFrame-like object, got {type(ragas_df)}"
            ) from exc

    non_metric_cols = {"question", "answer", "contexts", "ground_truth"}
    ragas_metric_cols = [col for col in ragas_df.columns if col not in non_metric_cols]

    if not ragas_metric_cols:
        return df, []

    ragas_metrics_df = ragas_df[ragas_metric_cols].reset_index(drop=True)
    merged_df = pd.concat([df.reset_index(drop=True), ragas_metrics_df], axis=1)

    return merged_df, ragas_metric_cols


def main() -> None:
    """Run evaluation on the configured gold set."""
    config = load_config()
    evaluation_dir = Path(config["paths"]["evaluation_dir"])
    gold_set_path = evaluation_dir / "gold_set.json"

    gold_set = load_gold_set(str(gold_set_path))
    results = [evaluate_example(example) for example in gold_set]
    summary = summarize_results(results)

    ragas_enabled = config.get("evaluation", {}).get("enable_ragas", False)
    ragas_summary: Optional[Dict[str, float]] = None
    ragas_output_path: Optional[Path] = None
    ragas_metric_cols: List[str] = []

    df = pd.DataFrame(results)

    if ragas_enabled:
        try:
            ragas_summary, ragas_df = run_ragas_evaluation(results)
            df, ragas_metric_cols = merge_ragas_per_question(df, ragas_df)

            ragas_output_path = evaluation_dir / "ragas_results.json"
            save_ragas_results(ragas_summary, ragas_output_path)

            ragas_per_question_path = evaluation_dir / "ragas_per_question.csv"
            ragas_df.to_csv(ragas_per_question_path, index=False)
            print(f"Saved per-question RAGAS results to: {ragas_per_question_path}")
        except Exception as exc:
            print("\n=== RAGAS Evaluation Skipped ===")
            print(f"Reason: {exc}")
            ragas_summary = None

    output_path = evaluation_dir / "evaluation_results.csv"
    df.to_csv(output_path, index=False)

    log_evaluation_summary(
        summary=summary,
        results=results,
        evaluation_csv_path=output_path,
        ragas_summary=ragas_summary,
        ragas_json_path=ragas_output_path,
    )

    print("\n=== Evaluation Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if ragas_summary:
        print("\n=== RAGAS Summary ===")
        for key, value in ragas_summary.items():
            print(f"{key}: {round(value, 4)}")

    print("\n=== Per-Question Results ===")
    display_cols = [
        "id",
        "document_hit",
        "page_hit",
        "token_overlap_score",
        "answer_pass",
        "latency_seconds",
    ]

    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if col in df.columns:
            display_cols.append(col)

    print(df[display_cols].to_string(index=False))

    print(f"\nSaved detailed results to: {output_path}")

    if ragas_output_path:
        print(f"Saved RAGAS summary to: {ragas_output_path}")


if __name__ == "__main__":
    main()