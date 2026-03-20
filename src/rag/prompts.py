"""Prompt templates for AlphaQuery."""

from __future__ import annotations

from typing import List


SYSTEM_PROMPT = """You are AlphaQuery, a financial document question-answering assistant.

Your job is to answer the user's question using only the retrieved document context provided.
Do not use outside knowledge.
If the answer is not supported by the retrieved context, say:
"I could not find enough evidence in the provided documents to answer this confidently."

Rules:
- Be concise, factual, and grounded.
- Do not invent facts.
- Prefer direct synthesis from the retrieved text.
- Use citations for key claims.
- Citation format must look like: [Document Name, p. X]
"""


def build_context_block(retrieved_docs: List) -> str:
    """Build prompt context from retrieved documents.

    Args:
        retrieved_docs: Retrieved LangChain documents.

    Returns:
        Concatenated formatted context string.
    """
    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):
        document_name = doc.metadata.get("document_name", "Unknown Document")
        page_number = doc.metadata.get("page_number", "?")
        content = doc.page_content.strip()

        context_parts.append(
            f"[Source {i}] {document_name}, page {page_number}\n{content}"
        )

    return "\n\n".join(context_parts)


def build_user_prompt(question: str, retrieved_docs: List) -> str:
    """Build the user prompt for grounded answer generation.

    Args:
        question: User question.
        retrieved_docs: Retrieved LangChain documents.

    Returns:
        Final user prompt string.
    """
    context_block = build_context_block(retrieved_docs)

    return f"""Question:
{question}

Retrieved context:
{context_block}

Instructions:
- Answer only from the retrieved context.
- If evidence is insufficient, say so clearly.
- Include citations inline for the main claims.
- Do not cite sources that are not in the retrieved context.
"""