# CS 288 Assignment 3 — EECS RAG System

A retrieval-augmented generation (RAG) system that answers factoid questions about UC Berkeley EECS using crawled web content. It combines sparse (BM25) and dense (FAISS) retrieval with Reciprocal Rank Fusion and an LLM reader.

---

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

---

## Architecture

```
┌────────────────────────────── Offline ──────────────────────────────┐
│                                                                     │
│  corpus.jsonl ──▶ Clean & Chunk ──┬──▶ BM25 Index ──┐               │
│                    (~500 chars)   │                 ├──▶ datastore/ │
│                                   └──▶ MiniLM ──▶ FAISS Index ──┘   │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────────────────── Online ───────────────────────────────┐
│                                                                     │
│  Question ──▶ LLM Query Expansion (3 phrasings)                     │
│                 │                                                   │
│                 ├──▶ BM25  (top 20 per query) ──┐                   │
│                 │                               ├──▶ RRF (top 20)   │
│                 └──▶ FAISS (top 20 per query) ──┘       │           │
│                                                         │           │
│                                          Cross-Encoder Rerank       │
│                                                (top 5)              │
│                                                  │                  │
│                                                 LLM ──▶ Answer      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Offline Phase — Building the Index

Run once to prepare the retrieval datastore. Pre-built files can be copied from another machine to skip these steps.

### 1. Crawl

```bash
python3 crawl_eecs.py
```

Recursively crawls the EECS website and writes `corpus.jsonl` — one record per page with `url`, `title`, and `text`.

To add individual pages without re-crawling:

```bash
python3 inject_url.py https://example.com/page
```

Fetches the URL (HTML or PDF), extracts text, deduplicates against the existing corpus, and appends a new entry to `corpus.jsonl`.

### 2. Build Index

```bash
python3 build_index.py
```

Reads `corpus.jsonl`, strips boilerplate, splits pages into ~500-character overlapping chunks, and builds two indices saved to `datastore/`:

| File | Contents |
|---|---|
| `chunks.json` | Raw chunk text and metadata |
| `bm25_index.pkl` | Sparse BM25 index for keyword matching |
| `faiss_index.bin` | Dense FAISS index (all-MiniLM-L6-v2, 384-dim) |
| `embedding_model/` | Model weights saved locally |

---

## Online Phase — Running the System

### Quick Start

If `datastore/` and `corpus.jsonl` already exist:

```bash
bash run.sh questions.txt predictions.txt
```

Reads one question per line, writes one answer per line.

### How Retrieval Works

For each question, `rag.py` runs a four-step pipeline:

1. **Query expansion** — The LLM generates 3 alternative phrasings of the question. All variants are used for retrieval, improving recall for ambiguous questions.
2. **Hybrid retrieval** — Each query variant is run against BM25 (top 20) and FAISS (top 20). BM25 catches exact keyword matches (names, course numbers); FAISS catches semantically similar passages. All results are merged with Reciprocal Rank Fusion (RRF), keeping the top 20 candidates.
3. **Cross-encoder reranking** — The top 20 candidates are reranked with a cross-encoder, which scores each (question, chunk) pair jointly for more precise relevance. The top 5 are kept.
4. **Answer generation** — The top 5 chunks are passed as context to an LLM via OpenRouter. The LLM extracts the shortest correct answer (1–10 words). The response is post-processed to strip quotes, prefixes, and trailing punctuation.

---

## Evaluation & Diagnostics

### Test the RAG system

```bash
python3 test_rag.py                                   # run on reference.jsonl (10 questions)
python3 test_rag.py --dataset qa_dataset.jsonl        # run on the full QA dataset
python3 test_rag.py --question "Who is Dan Klein?"    # single question
```

Reports per-question diagnostics (retrieved chunks, retrieval recall, error classification) and aggregate Exact Match / F1 metrics.

### Diagnose failures

```bash
python3 diagnose.py
```

Runs the full pipeline on a QA dataset and categorises each result as one of:

| Label | Meaning |
|---|---|
| `CORRECT` | Answer matches gold |
| `PARTIAL` | Partial F1 match |
| `READER_FAIL` | Correct chunk retrieved but LLM answered wrong |
| `RERANK_MISS` | Correct chunk retrieved but dropped by reranker |
| `RETRIEVAL_MISS` | Correct chunk never retrieved |

Prints per-question details and a summary with actionable recommendations.

### Local batch evaluation

```bash
python3 evaluate_local.py
```

Runs EM/F1 evaluation over a predictions file without calling the LLM.

---

## QA Dataset

`qa_dataset.jsonl` was written **by hand** — questions and gold answers were manually authored and verified against the EECS website. This replaces an earlier LLM-generated dataset to ensure factual accuracy.

---

## Supporting Scripts

| Script | Purpose |
|---|---|
| `llm.py` | Thin wrapper around the OpenRouter API |
| `generate_qa.py` | (Legacy) Generate QA pairs from the corpus via LLM |
| `measure_iaa.py` | Measure inter-annotator agreement on the QA dataset |
| `evaluate-v1.1.py` | Official Gradescope evaluation script (EM/F1) |
| `create_submission_zip.sh` | Package files for Gradescope submission |

---

## Submission

```bash
bash create_submission_zip.sh
```

Produces `submission.zip` ready for Gradescope upload.
