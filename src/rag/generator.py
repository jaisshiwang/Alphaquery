"""LLM answer generation utilities."""

from __future__ import annotations

from typing import Tuple

from src.providers.llm import get_llm_client
from src.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from src.utils.config import load_config


def generate_answer(question: str, retrieved_docs: list) -> Tuple[str, dict]:
    """Generate a grounded answer from retrieved documents.

    Args:
        question: User question.
        retrieved_docs: Retrieved LangChain documents.

    Returns:
        Tuple of generated answer text and raw model response metadata.
    """
    config = load_config()
    llm_config = config["llm"]
    client = get_llm_client(llm_config)

    user_prompt = build_user_prompt(question, retrieved_docs)

    response = client.chat.completions.create(
        model=llm_config["model_name"],
        temperature=llm_config["temperature"],
        max_tokens=llm_config["max_tokens"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content.strip()
    metadata = {
        "model_name": llm_config["model_name"],
        "provider": llm_config["provider"],
        "usage": getattr(response, "usage", None),
    }

    return answer, metadata