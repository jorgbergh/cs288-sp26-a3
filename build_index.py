#!/usr/bin/env python3
"""
Offline index builder for the RAG system.
Reads corpus.jsonl, cleans text, chunks documents, builds BM25 and FAISS indices.
Saves everything to datastore/ for use at runtime.
"""

import json
import os
import pickle
import re
import sys

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CORPUS_FILE = "corpus.jsonl"
DATASTORE_DIR = "datastore"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Corpus cleaning
# ---------------------------------------------------------------------------

BREADCRUMB_RE = re.compile(
    r"^(Home\n/\n.*?\n)+", re.MULTILINE
)

BOILERPLATE_LINES = {
    "eecs at uc berkeley",
    "eecs at berkeley",
    "accessibility",
    "nondiscrimination",
    "privacy",
    "contact",
    "© 2023 uc regents",
    "© 2024 uc regents",
    "© 2025 uc regents",
    "give to eecs",
    "berkeley eecs on twitter",
    "berkeley eecs on instagram",
    "berkeley eecs on linkedin",
    "berkeley eecs on youtube",
    "learn more about the campaign for berkeley and graduate fellowships.",
}

SIDEBAR_SECTIONS = [
    "About\nHistory\nDiversity\nVisiting\nSpecial Events",
    "People\nDirectory\nLeadership\nFaculty\nStaff\nStudents\nAlumni",
    "Connect\nSupport Us\nK-12 Outreach\nFaculty Positions\nStaff Positions\nContact",
    "Academics\nUndergrad Admissions & Programs\nGraduate Admissions & Programs\nCourses",
    "Resources\nRoom Reservations\nMy EECS Info",
    "Research\nAreas\nCenters & Labs\nProjects\nTechnical Reports\nPhD Dissertations",
    "Industry\nRecruit Students\nCorporate Access",
]


def clean_text(text: str, title: str) -> str:
    """Remove breadcrumbs, boilerplate, and sidebar navigation from text."""
    # Strip breadcrumb navigation at the start
    text = BREADCRUMB_RE.sub("", text)

    # Remove www2 sidebar navigation blocks
    for block in SIDEBAR_SECTIONS:
        text = text.replace(block, "")

    # Remove boilerplate lines
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in BOILERPLATE_LINES:
            continue
        if stripped == "":
            continue
        cleaned.append(stripped)
    text = "\n".join(cleaned)

    # Remove duplicate title from body if it appears at the very start
    if title and text.startswith(title):
        text = text[len(title):].lstrip("\n")

    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks by character count, breaking at line boundaries."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a newline near the end
        if end < len(text):
            newline_pos = text.rfind("\n", start + chunk_size // 2, end + 50)
            if newline_pos > start:
                end = newline_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


# ---------------------------------------------------------------------------
# BM25 tokenizer
# ---------------------------------------------------------------------------

TOKENIZE_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list:
    return TOKENIZE_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATASTORE_DIR, exist_ok=True)

    # --- Load corpus ---
    print("Loading corpus...", file=sys.stderr)
    docs = []
    with open(CORPUS_FILE) as f:
        for line in f:
            docs.append(json.loads(line.strip()))
    print(f"  Loaded {len(docs)} documents", file=sys.stderr)

    # --- Clean and chunk ---
    print("Cleaning and chunking...", file=sys.stderr)
    chunks = []  # list of {url, title, text}
    for doc in tqdm(docs, desc="Chunking", file=sys.stderr):
        cleaned = clean_text(doc["text"], doc.get("title", ""))
        if len(cleaned) < 30:
            continue

        doc_chunks = chunk_text(cleaned)
        for chunk_text_str in doc_chunks:
            chunks.append({
                "url": doc["url"],
                "title": doc.get("title", ""),
                "text": chunk_text_str,
            })

    print(f"  Total chunks: {len(chunks)}", file=sys.stderr)

    # Save chunk metadata
    chunks_path = os.path.join(DATASTORE_DIR, "chunks.json")
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"  Saved chunks to {chunks_path}", file=sys.stderr)

    # --- Build BM25 index ---
    print("Building BM25 index...", file=sys.stderr)
    tokenized_chunks = [tokenize(c["text"]) for c in tqdm(chunks, desc="Tokenizing", file=sys.stderr)]
    bm25 = BM25Okapi(tokenized_chunks)

    bm25_path = os.path.join(DATASTORE_DIR, "bm25_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  Saved BM25 index to {bm25_path}", file=sys.stderr)

    # --- Build dense (FAISS) index ---
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...", file=sys.stderr)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Save model locally for offline use on Gradescope
    model_save_path = os.path.join(DATASTORE_DIR, "embedding_model")
    model.save(model_save_path)
    print(f"  Saved embedding model to {model_save_path}", file=sys.stderr)

    print("Encoding chunks...", file=sys.stderr)
    chunk_texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        chunk_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype(np.float32)

    print("Building FAISS index...", file=sys.stderr)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss_path = os.path.join(DATASTORE_DIR, "faiss_index.bin")
    faiss.write_index(index, faiss_path)
    print(f"  Saved FAISS index ({index.ntotal} vectors, {dim}d) to {faiss_path}", file=sys.stderr)

    # --- Summary ---
    print(f"\nBuild complete!", file=sys.stderr)
    print(f"  Documents: {len(docs)}", file=sys.stderr)
    print(f"  Chunks: {len(chunks)}", file=sys.stderr)
    print(f"  BM25 index: {bm25_path}", file=sys.stderr)
    print(f"  FAISS index: {faiss_path}", file=sys.stderr)
    print(f"  Embedding model: {model_save_path}", file=sys.stderr)
    print(f"  Chunks metadata: {chunks_path}", file=sys.stderr)

    # Print datastore size
    total_size = 0
    for root, dirs, files in os.walk(DATASTORE_DIR):
        for fname in files:
            total_size += os.path.getsize(os.path.join(root, fname))
    print(f"  Total datastore size: {total_size / 1024 / 1024:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
