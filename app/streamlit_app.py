"""Streamlit app for AlphaQuery."""

from __future__ import annotations

import streamlit as st

from src.rag.chain import answer_question
from src.utils.config import load_config


def main() -> None:
    """Run the Streamlit application."""
    config = load_config()

    st.set_page_config(page_title="AlphaQuery", layout="wide")
    st.title("AlphaQuery")
    st.caption("RAG-powered financial document assistant for investment outlook reports")

    with st.sidebar:
        st.header("Retrieval Settings")

        top_k = st.slider(
            "Top-K chunks",
            min_value=1,
            max_value=8,
            value=config["retrieval"]["top_k"],
            step=1,
        )

        st.header("Metadata Filters")

        publisher = st.selectbox(
            "Publisher",
            options=["All", "BlackRock", "Fidelity", "PIMCO", "Vanguard"],
            index=0,
        )

        year = st.selectbox(
            "Year",
            options=["All", 2024],
            index=0,
        )

        document_name = st.selectbox(
            "Document",
            options=[
                "All",
                "BlackRock 2024 Global Outlook",
                "Fidelity International Outlook 2024",
                "PIMCO_Cyclical_Outlook_Wilding_Balls_Jan2024",
                "Vanguard Outlook 2024",
            ],
            index=0,
        )

    question = st.text_area(
        "Ask a question about the financial outlook documents",
        placeholder="Example: What is Fidelity's base case for 2024?",
        height=120,
    )

    ask_button = st.button("Run Query", type="primary")

    if ask_button:
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving evidence and generating answer..."):
            result = answer_question(
                question=question.strip(),
                top_k=top_k,
                publisher=None if publisher == "All" else publisher,
                year=None if year == "All" else year,
                document_name=None if document_name == "All" else document_name,
            )

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Citations")
        if result["citations"]:
            for citation in result["citations"]:
                st.markdown(f"- {citation}")
        else:
            st.info("No citations available.")

        st.subheader("Run Metadata")
        st.write(
            {
                "latency_seconds": result["latency_seconds"],
                "llm_metadata": result["llm_metadata"],
                "retrieved_chunk_count": len(result["retrieved_docs"]),
            }
        )

        if config["app"]["show_retrieved_chunks"]:
            with st.expander("Retrieved Chunks"):
                for i, doc in enumerate(result["retrieved_docs"], start=1):
                    st.markdown(f"### Chunk {i}")
                    st.json(doc.metadata)
                    st.write(doc.page_content)


if __name__ == "__main__":
    main()