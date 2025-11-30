# 🧠 Advanced RAG System — Multi-Retriever, Multi-Rewriting, Evaluation, Streamlit UI

Built with OpenRouter (4o-mini), ChromaDB, BM25, HyDE, Multi-Query, Self-Ask, RRF Fusion

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Running the System](#running-the-system)
- [Features Breakdown](#features-breakdown)
- [Why This System](#why-this-system)

## Overview

This project implements a fully featured Advanced Retrieval-Augmented Generation (RAG) system, following modern industry practices (2024–2025), including:

- **Query Routing** — decides the rewriting strategy
- **Query Rewriting**
    - Multi-Query Rewriting
    - HyDE (Hypothetical Document Expansion)
    - Self-Ask (Step-by-step decomposition)
- **Chunking & Indexing Pipeline** — PDF extraction → cleaning → overlap chunking
- **Dual Index** — dense (embeddings + ChromaDB) & lexical (BM25)
- **Retrieval Controller** — dynamic retrieval depending on rewriting type
- **Fusion Strategies** — lightweight concatenation & RRF (Reciprocal Rank Fusion)
- **Grounded Answer Generation** — OpenRouter LLM (GPT-4o-mini)
- **Evaluation Module** — relevance, support, final score
- **Streamlit Interface** — interactive UI with answer, rewriting method, retrieved chunks, evaluation

This system allows you to upload PDFs, index them, and ask complex questions with advanced semantic and lexical retrieval.

## Project Structure

```
project/
├── data/                  # PDF files, chunked data, vector store, bm25 index
├── src/
│   ├── chunking/
│   │   ├── chunker.py     # PDF loading → text cleaning → chunking
│   │   ├── indexer.py     # embeddings via OpenRouter → ChromaDB + BM25
│   │   └── __init__.py
│   ├── rewriting/
│   │   ├── router.py      # chooses rewriting method
│   │   ├── multi_query.py
│   │   ├── hyde.py
│   │   ├── self_ask.py
│   │   ├── apply_method.py
│   │   └── __init__.py
│   ├── retrieval/
│   │   ├── dense_retriever.py
│   │   ├── bm25_retriever.py
│   │   ├── rr_fusion.py
│   │   ├── retrieval_controller.py
│   │   └── __init__.py
│   ├── generation/
│   │   ├── generator.py
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   └── __init__.py
│   ├── pipeline_rag.py    # Orchestrator: rewriting → retrieve → generate → evaluate
│   └── utils/
├── cli.py                 # Command-line interface
├── app.py                 # Streamlit web interface
├── config.yaml
└── README.md
```

## Running the System

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Chunk PDFs & Build Indexes
Put your PDFs in `data/pdfs/`, then run:
```bash
python test_chunker.py
python test_indexer.py
```

### 3️⃣ Run CLI
```bash
python cli.py --query "What is cognitive dissonance?"
```

### 4️⃣ Run Streamlit Interface
```bash
streamlit run app.py
```

## Features Breakdown

### 🔹 Query Router
Uses OpenRouter to decide whether your question needs:
- Multi-query expansion
- HyDE synthetic document
- Self-ask decomposition
- No rewriting (simple question)

### 🔹 Query Rewriting

| Method | When Used | Purpose |
|--------|-----------|---------|
| HyDE | Broad conceptual questions | Create a synthetic document → embed → dense search |
| Multi-query Rewriting | Ambiguous questions | 5 paraphrases → multi-search → RRF fusion |
| Self-Ask | Multi-step reasoning | Expand question into sub-questions |
| None | Simple factual query | Direct retrieval |

### 🔹 Retrieval Controller

| Rewriting Type | Retrieval Method |
|----------------|------------------|
| HyDE | Dense retrieval |
| Multi-query | Multi-dense + RRF |
| Self-Ask | Multi-retrieval (dense+BM25) + RRF |
| None | Hybrid retrieval |

### 🔹 Evaluation

Computes:
- **Similarity score** — cosine between answer embedding & chunk embeddings
- **Support** — amount of information
- **Final weighted score** — displayed in the Streamlit app

## Why This System?

Because modern RAG (Microsoft 2024, Meta 2024, OpenAI 2025) uses:

- Hybrid retrieval (dense + sparse)
- Query understanding (rewriting)
- Reasoning-based decomposition (self-ask)
- Fusion-based document ranking
- Evaluation heuristics for quality control

This project matches real-world production RAG architectures.
