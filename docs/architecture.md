# AlphaQuery Architecture

```text
PDF Documents
   ↓
Bronze Layer
- page-level extraction
- raw metadata

   ↓
Silver Layer
- cleaned text
- recursive chunking
- metadata enrichment

   ↓
Gold Layer
- embeddings
- Chroma vector store

   ↓
Retriever
- semantic similarity
- optional metadata filters

   ↓
LLM Generator
- grounded prompt
- answer synthesis
- citations

   ↓
Evaluation + Monitoring
- gold set evaluation
- ragas evaluation
- MLflow logging