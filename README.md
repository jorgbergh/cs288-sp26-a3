# CS 288 Assignment 3 — EECS RAG System

A retrieval-augmented generation system that answers factoid questions about UC Berkeley EECS using crawled web pages.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your OpenRouter API key:

```
OPENROUTER_API_KEY=your_key_here
```

## Quick Start (with pre-built data)

If you already have the `datastore/` folder and `corpus.jsonl`, skip straight to running:

```bash
bash run.sh questions.txt predictions.txt
```

The `datastore/` folder must contain:

```
datastore/
  chunks.json
  bm25_index.pkl
  faiss_index.bin
  embedding_model/     # BAAI/bge-small-en-v1.5 saved locally
```

You can copy these files in from another machine instead of rebuilding them.

## Full Pipeline (from scratch)

### 1. Crawl the EECS website

```bash
python3 crawl_eecs.py
```

Produces `corpus.jsonl` — one JSON object per page with `url`, `title`, and `text` fields.

### 2. Build the retrieval index

```bash
python3 build_index.py
```

Reads `corpus.jsonl`, cleans the text, chunks it into ~500-char passages, and builds BM25 + FAISS indices. Saves everything to `datastore/`.

### 3. Run the RAG system

```bash
bash run.sh questions.txt predictions.txt
```

Reads one question per line from the input file, writes one answer per line to the output file.

## How It Works

```
┌─────────────────────────── Offline (build_index.py) ───────────────────────────┐
│                                                                                │
│  corpus.jsonl ──▶ Clean & Chunk ──┬──▶ BM25 Index ──────┐                      │
│                   (~500 chars)    │                      ├──▶ datastore/        │
│                                   └──▶ BGE Embeddings ──▶ FAISS Index ──┘      │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────-┘

┌─────────────────────────── Runtime (rag.py) ───────────────────────────────────┐
│                                                                                │
│  Question ──┬──▶ BM25 (top 20) ──┐                                             │
│             │                    ├──▶ RRF Merge (top 5) ──▶ LLM ──▶ Answer     │
│             └──▶ FAISS (top 20) ─┘                                             │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────-┘
```

The system has two phases:

**Offline** — `build_index.py` reads the crawled corpus, strips boilerplate (breadcrumbs, nav elements), and splits each page into ~500-character chunks with overlap. Each chunk is indexed in two ways: a BM25 sparse index for keyword matching, and a FAISS dense index using `BAAI/bge-small-en-v1.5` embeddings (384-dim, ~130MB) for semantic similarity.

**Runtime** — For each question, `rag.py` queries both indices in parallel. BM25 catches exact keyword matches (e.g. names, course numbers). FAISS catches semantically similar passages even when wording differs. The two ranked lists are merged using Reciprocal Rank Fusion (RRF) and the top 5 chunks are kept.

These chunks are passed as context to `meta-llama/llama-3.1-8b-instruct` via OpenRouter. The prompt instructs the LLM to extract the shortest possible answer (1–10 words) directly from the context. The raw response is post-processed to strip quotes, prefixes, and trailing punctuation.

## Testing & Evaluation

```bash
python3 test_rag.py                                    # run on reference.jsonl (10 questions)
python3 test_rag.py --dataset qa_dataset.jsonl         # run on the full QA dataset
python3 test_rag.py --question "Who is Dan Klein?"     # test a single question
```

Shows per-question diagnostics (retrieved chunks, retrieval recall, error classification) and aggregate EM/F1 metrics.

## Other Scripts

| Script | Purpose |
|---|---|
| `generate_qa.py` | Generate QA pairs from the corpus using LLMs |
| `measure_iaa.py` | Measure inter-annotator agreement on the QA dataset |
| `evaluate_local.py` | Batch evaluation with EM/F1 metrics |
| `create_submission_zip.sh` | Package files for Gradescope submission |

## Submission

```bash
bash create_submission_zip.sh
```

Produces `submission.zip` ready for Gradescope upload.
