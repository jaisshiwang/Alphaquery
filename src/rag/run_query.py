"""Simple local query runner for AlphaQuery."""

from __future__ import annotations

from src.rag.chain import answer_question


def main() -> None:
    """Run one example query locally."""
    question = "What is Fidelity's base case for 2024?"

    result = answer_question(question=question, publisher="Fidelity")

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nCITATIONS:")
    for citation in result["citations"]:
        print(f"- {citation}")

    print("\nLATENCY:")
    print(result["latency_seconds"], "seconds")

    print("\nRETRIEVED CHUNKS:")
    for i, doc in enumerate(result["retrieved_docs"], start=1):
        print(f"\n--- Chunk {i} ---")
        print(doc.metadata)
        print(doc.page_content[:400])
        
    print("\n Complete result object:")
    for key, value in result.items():
        if key != "retrieved_docs":  # Avoid printing full retrieved docs again
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()