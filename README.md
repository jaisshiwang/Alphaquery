# AlphaQuery

AlphaQuery is a lightweight Retrieval-Augmented Generation (RAG) system for answering questions over financial outlook documents. It was built as a practical MVP for document Q&A across a small corpus of public investment reports from firms such as BlackRock, Fidelity, PIMCO, and Vanguard.

The system ingests PDF documents, extracts and chunks them into a medallion-style pipeline, indexes them in a vector store, retrieves relevant evidence for a user question, and generates a grounded answer with source citations.

---
## Demo

Example query:
> What is Fidelity's base case for 2024?

Answer:
> Fidelity's base case for 2024 is a cyclical recession.

Citations:
- [Fidelity International Outlook 2024, p. 5]

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

### Medallion layers

Bronze
	•	raw extracted page text
	•	document metadata
	•	extraction provenance

Silver
	•	cleaned page text
	•	recursive chunking with overlap
	•	metadata attached to each chunk

Gold
	•	embedded retrieval-ready chunk objects
	•	persisted Chroma vector store
	•	used at query time for retrieval

## 4. Repository Structure
alphaquery/
├── app/
├── configs/
├── data/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   ├── documents/
│   └── evaluation/
├── src/
│   ├── ingestion/
│   ├── providers/
│   ├── rag/
│   ├── evaluation/
│   ├── monitoring/
│   └── utils/
├── tests/
├── .env.example
├── Makefile
├── requirements.txt
└── README.md

## 5. Setup Instructions
Environment

Tested on Python 3.9.12.

Clone and set up

git clone https://github.com/jaisshiwang/Alphaquery
cd alphaquery

If required :
(Add API key
Create a .env file in the project root:
GROQ_API_KEY=your_actual_key_here

Configure the LLM

The system uses a config-driven LLM provider interface. The default configuration is:
llm:
  provider: openai_compatible
  model_name: llama-3.1-8b-instant
  base_url: https://api.groq.com/openai/v1
  api_key_env: GROQ_API_KEY
  temperature: 0.0
  max_tokens: 700)

make bootstrap
make run


## 6. How to Run
Build the pipeline
bash:
make bootstrap

Run the app
bash:
make run

Reun evaluation:
bash:
make eval

Launch MLflowUIO
bash:
makemlflow-ui

## 7. Design Desicions

Why dense semantic retrieval?

The MVP uses dense semantic retrieval with sentence-transformers/all-MiniLM-L6-v2 embeddings and Chroma vector similarity search.

This was chosen because:
	•	the corpus is small
	•	semantic retrieval is simple and fast to implement
	•	it works well for concept-based questions and paraphrases
	•	it provides a strong baseline before introducing more complexity

A future improvement would be hybrid retrieval to better handle exact financial terminology, headings, and numerical queries.

Why this chunking strategy?

The documents are narrative, section-based financial outlook PDFs. I used medium-sized recursive chunks with overlap to balance retrieval precision with context preservation.

This works well because:
	•	too-small chunks lose supporting context
	•	too-large chunks increase retrieval noise
	•	overlap reduces information loss across chunk boundaries
	•	page-bounded chunking preserves source citation accuracy

Why metadata-rich chunks?

Each chunk stores:
	•	document name
	•	publisher
	•	page number
	•	year
	•	chunk id
	•	document type

This supports:
	•	metadata filtering
	•	source citations
	•	retrieval debugging
	•	clearer evaluation

Why medallion architecture?

Even for a small RAG project, medallion layers make the pipeline easier to debug and evolve:
	•	Bronze isolates extraction quality
	•	Silver isolates chunking and metadata logic
	•	Gold isolates retrieval-serving artifacts

This makes reprocessing simpler when extraction, chunking, or embedding strategies change.

Why a config-driven provider layer?

The LLM provider is configurable via:
	•	provider name
	•	model name
	•	base URL
	•	API key environment variable

This allows the system to switch between Groq, Gemini, OpenAI-compatible endpoints, or local providers with minimal changes.

## 8. Evaluation
A small gold set of 5 question-answer pairs was created against the selected financial document corpus.

The evaluation checks both retrieval quality and answer quality.

Metrics used
	•	Document Hit Rate: whether the expected document was retrieved
	•	Page Hit Rate: whether an expected supporting page was retrieved
	•	Token Overlap Score: simple lexical overlap between expected and generated answers
	•	Answer Pass Rate: whether the answer exceeded a token-overlap threshold

Latest results
	•	Document Hit Rate: 1.00
	•	Page Hit Rate: 0.80
	•	Answer Pass Rate: 1.00
	•	Average Token Overlap: 0.675
	•	Average Latency: 1.95s

Interpretation

The retriever consistently identified the correct source document across the gold set. Page-level precision was also strong, though not perfect, indicating room for retrieval refinement. The generated answers performed well on the gold set, but the current evaluation remains intentionally lightweight and does not fully verify faithfulness or citation correctness.

⸻

## 9. Observability and Monitoring

MLflow is used as a lightweight LLMOps / observability layer.

Query-time logging

For each query, the system logs:
	•	question
	•	applied filters
	•	model name and provider
	•	latency
	•	retrieved chunk count
	•	retrieved document names
	•	retrieved pages
	•	answer text
	•	citations

Evaluation logging

For benchmark runs, the system logs:
	•	evaluation summary metrics
	•	detailed per-question results

This makes it easier to compare runs and debug regressions in retrieval or generation behavior.

⸻

## 10. Production Thinking

Observability

In a production system, I would expand monitoring to include:
	•	token usage
	•	API errors and retries
	•	retrieval scores
	•	prompt versions
	•	answer quality dashboards
	•	trace IDs per request

Cost control

Current cost control choices:
	•	local embeddings to avoid embedding API cost
	•	small free-tier hosted LLM
	•	limited top-k retrieval
	•	concise prompts

Future improvements:
	•	query caching
	•	model routing by query complexity
	•	prompt truncation and token budgets

Error handling

The current system includes lightweight handling for:
	•	missing API keys
	•	empty retrieval results
	•	malformed PDFs
	•	local dependency issues during setup

In production, I would add:
	•	retry policies
	•	timeout handling
	•	fallback models
	•	structured error logging

Security

For this local MVP:
	•	secrets are stored in .env
	•	no API keys are hardcoded

In production, I would additionally implement:
	•	secret managers
	•	permission-aware retrieval
	•	prompt injection hardening
	•	sensitive content logging controls
	•	document-level access control

Scaling to 10,000 documents

At larger scale, I would likely:
	•	separate ingestion and serving services
	•	move to a production-grade vector database
	•	add hybrid retrieval and reranking
	•	implement document versioning and re-index workflows
	•	support permission-aware retrieval
	•	build continuous evaluation and monitoring pipelines

⸻

## 11. Limitations

Current limitations include:
	•	dense retrieval only; no hybrid lexical retrieval yet
	•	citation attribution is chunk-level, not span-level
	•	evaluation is lightweight and not a full faithfulness benchmark
	•	chunking does not yet use robust section-header detection
	•	some low-information chunks such as contents pages may still be indexed

⸻

## 12. Next Steps

Planned improvements:
	•	hybrid retrieval (vector + BM25)
	•	reranking for improved page-level precision
	•	stronger citation attribution
	•	removal of low-information chunks during ingestion
	•	LLM-as-judge evaluation for faithfulness and relevance
	•	improved section-aware chunking
	•	Dockerized setup for reproducibility