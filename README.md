# AlphaQuery

AlphaQuery is a lightweight Retrieval-Augmented Generation (RAG) system for answering questions over financial outlook documents. It was built as a practical MVP for document Q&A across a small corpus of public investment reports from firms such as BlackRock, Fidelity, PIMCO, and Vanguard.

The system ingests PDF documents, extracts and chunks them into a medallion-style pipeline, indexes them in a vector store, retrieves relevant evidence for a user question, and generates a grounded answer with source citations.

---

## 1. Overview

This project demonstrates a small but complete financial document Q&A assistant with:

- PDF ingestion and page-level extraction
- Chunking and metadata enrichment
- Dense semantic retrieval using vector embeddings
- Grounded answer generation with citations
- Metadata filtering by publisher, year, and document
- Streamlit-based user interface
- Lightweight evaluation on a gold question set
- MLflow-based query and evaluation logging

---

## 2. Project Goals

The goal was to build a simple, explainable, and reproducible RAG MVP rather than over-engineer a production system.

The design priorities were:

- grounded answers over creative generation
- clean source traceability
- modular architecture
- easy local setup
- simple but honest evaluation
- clear room for future improvements such as hybrid retrieval and stronger citation attribution

---

## 3. Architecture

### High-level flow

```text
Financial PDFs
   ↓
Bronze: raw page-level extraction
   ↓
Silver: cleaned, chunked, metadata-rich records
   ↓
Gold: embedded chunks stored in Chroma
   ↓
Retriever (semantic similarity + optional metadata filters)
   ↓
LLM prompt with retrieved context
   ↓
Grounded answer + citations
   ↓
Evaluation + MLflow logging