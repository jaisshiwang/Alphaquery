"""Optional RAGAS evaluator for AlphaQuery."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_recall, faithfulness

from src.providers.embeddings import get_embeddings
from src.utils.config import load_config

load_dotenv()


def build_ragas_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert evaluated examples into RAGAS dataset rows."""
    rows: List[Dict[str, Any]] = []

    for result in results:
        rows.append(
            {
                "question": result["question"],
                "answer": result["predicted_answer"],
                "contexts": result["retrieved_contexts"],
                "ground_truth": result["expected_answer"],
            }
        )

    return rows


def get_ragas_judge_llm(config: Dict[str, Any]) -> LangchainLLMWrapper:
    """Create wrapped LangChain LLM for RAGAS."""
    llm_config = config["llm"]
    api_key = os.getenv(llm_config["api_key_env"])

    if not api_key:
        raise ValueError(
            f"Missing API key in environment variable: {llm_config['api_key_env']}"
        )

    judge_llm = ChatOpenAI(
        model=llm_config["model_name"],
        api_key=api_key,
        base_url=llm_config["base_url"],
        temperature=0.0,
    )

    return LangchainLLMWrapper(judge_llm)


def get_ragas_embeddings(config: Dict[str, Any]) -> LangchainEmbeddingsWrapper:
    """Create wrapped embeddings for RAGAS."""
    embeddings = get_embeddings(config["embeddings"])
    return LangchainEmbeddingsWrapper(embeddings)


def run_ragas_evaluation(
    results: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Run RAGAS evaluation and return summary + per-question dataframe."""
    config = load_config()
    rows = build_ragas_rows(results)
    dataset = Dataset.from_list(rows)

    ragas_result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            context_recall,
        ],
        llm=get_ragas_judge_llm(config),
        embeddings=get_ragas_embeddings(config),
    )

    # Convert safely to pandas
    if hasattr(ragas_result, "to_pandas"):
        df = ragas_result.to_pandas()
    elif isinstance(ragas_result, pd.DataFrame):
        df = ragas_result
    else:
        # last-resort normalization
        try:
            df = pd.DataFrame(ragas_result)
        except Exception as exc:
            raise TypeError(
                f"Unable to convert RAGAS result to DataFrame. Got type: {type(ragas_result)}"
            ) from exc

    summary: Dict[str, float] = {}
    non_metric_cols = {"question", "answer", "contexts", "ground_truth"}

    for column in df.columns:
        if column in non_metric_cols:
            continue
        try:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if not series.empty:
                summary[column] = float(series.mean())
        except Exception:
            continue

    return summary, df


def save_ragas_results(results: Dict[str, float], output_path: Path) -> None:
    """Save aggregated RAGAS results to JSON."""
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)