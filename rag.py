#!/usr/bin/env python3
"""
RAG pipeline for UC Berkeley EECS question answering.
Loads pre-built indices from datastore/, retrieves relevant passages,
and generates short answers via LLM.

Usage: python3 rag.py <questions_txt_path> <predictions_out_path>
"""

import json
import os
import pickle
import re
import sys
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load .env if present (for local development)
def _load_dotenv(path=".env"):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass

_load_dotenv()

from llm import call_llm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASTORE_DIR = os.path.join(SCRIPT_DIR, "datastore")

BM25_TOP_K = 20
DENSE_TOP_K = 20
FINAL_TOP_K = 7
RRF_K = 60  # Reciprocal Rank Fusion constant

LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"
LLM_MAX_TOKENS = 32
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT = 20

FALLBACK_ANSWER = "unknown"

SYSTEM_PROMPT = (
    "You are a factoid QA system for UC Berkeley EECS. "
    "Given context passages, answer the question with the shortest possible answer "
    "(1-10 words). Extract the answer verbatim from the context when possible. "
    "Output ONLY the answer, nothing else. No explanations, no full sentences."
)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOKENIZE_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list:
    return TOKENIZE_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# RAG System
# ---------------------------------------------------------------------------

class RAGSystem:
    def __init__(self):
        print("Loading RAG system...", file=sys.stderr)
        t0 = time.time()

        # Load chunks metadata
        chunks_path = os.path.join(DATASTORE_DIR, "chunks.json")
        with open(chunks_path) as f:
            self.chunks = json.load(f)

        # Load BM25 index
        bm25_path = os.path.join(DATASTORE_DIR, "bm25_index.pkl")
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

        # Load FAISS index
        faiss_path = os.path.join(DATASTORE_DIR, "faiss_index.bin")
        self.faiss_index = faiss.read_index(faiss_path)

        # Load embedding model (local copy or download from HuggingFace)
        model_path = os.path.join(DATASTORE_DIR, "embedding_model")
        if os.path.exists(model_path):
            self.embed_model = SentenceTransformer(model_path)
        else:
            print("  Local model not found, downloading BAAI/bge-small-en-v1.5...", file=sys.stderr)
            self.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            self.embed_model.save(model_path)

        elapsed = time.time() - t0
        print(f"  Loaded in {elapsed:.1f}s ({len(self.chunks)} chunks)", file=sys.stderr)

    def retrieve_bm25(self, query: str, top_k: int = BM25_TOP_K) -> list:
        """Retrieve top-k chunks using BM25."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def retrieve_dense(self, query: str, top_k: int = DENSE_TOP_K) -> list:
        """Retrieve top-k chunks using dense (FAISS) retrieval."""
        query_with_prefix = BGE_QUERY_PREFIX + query
        query_vec = self.embed_model.encode(
            [query_with_prefix], normalize_embeddings=True
        ).astype(np.float32)
        scores, indices = self.faiss_index.search(query_vec, top_k)
        return [(int(indices[0][i]), float(scores[0][i]))
                for i in range(len(indices[0])) if indices[0][i] >= 0]

    def hybrid_retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list:
        """Combine BM25 and dense retrieval using Reciprocal Rank Fusion."""
        bm25_results = self.retrieve_bm25(query)
        dense_results = self.retrieve_dense(query)

        rrf_scores = {}

        for rank, (idx, _score) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank + 1)

        for rank, (idx, _score) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (RRF_K + rank + 1)

        sorted_indices = sorted(rrf_scores.items(), key=lambda x: -x[1])
        return [idx for idx, _ in sorted_indices[:top_k]]

    def build_prompt(self, question: str, chunk_indices: list) -> str:
        """Build the user prompt with retrieved context."""
        context_parts = []
        for i, idx in enumerate(chunk_indices, 1):
            chunk = self.chunks[idx]
            context_parts.append(f"[{i}] {chunk['text']}")

        context = "\n\n".join(context_parts)
        return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    def answer_question(self, question: str) -> str:
        """Full RAG pipeline for a single question."""
        # Retrieve
        chunk_indices = self.hybrid_retrieve(question)

        if not chunk_indices:
            return FALLBACK_ANSWER

        # Build prompt
        prompt = self.build_prompt(question, chunk_indices)

        # Generate
        try:
            answer = call_llm(
                query=prompt,
                system_prompt=SYSTEM_PROMPT,
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                timeout=LLM_TIMEOUT,
            )
        except Exception as e:
            print(f"  LLM error: {e}", file=sys.stderr)
            return FALLBACK_ANSWER

        return self.postprocess(answer)

    @staticmethod
    def postprocess(answer: str) -> str:
        """Clean up the LLM output to produce a concise answer."""
        answer = answer.strip()
        # Remove trailing period
        if answer.endswith("."):
            answer = answer[:-1].strip()
        # Remove any newlines
        answer = answer.replace("\n", " ")
        # Remove common LLM prefixes
        for prefix in ["Answer:", "The answer is", "A:"]:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        # Remove surrounding quotes
        if len(answer) > 2 and answer[0] in ('"', "'") and answer[-1] == answer[0]:
            answer = answer[1:-1]
        return answer if answer else FALLBACK_ANSWER


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 rag.py <questions_txt_path> <predictions_out_path>", file=sys.stderr)
        sys.exit(1)

    questions_path = sys.argv[1]
    predictions_path = sys.argv[2]

    # Load questions
    with open(questions_path) as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(questions)} questions from {questions_path}", file=sys.stderr)

    # Initialize RAG system
    rag = RAGSystem()

    # Answer each question
    predictions = []
    for i, question in enumerate(tqdm(questions, desc="Answering", file=sys.stderr)):
        try:
            answer = rag.answer_question(question)
        except Exception as e:
            print(f"  Error on Q{i+1}: {e}", file=sys.stderr)
            answer = FALLBACK_ANSWER
        predictions.append(answer)

    # Write predictions
    with open(predictions_path, "w") as f:
        for pred in predictions:
            f.write(pred + "\n")

    print(f"\nWrote {len(predictions)} predictions to {predictions_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
