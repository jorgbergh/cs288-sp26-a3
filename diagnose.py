#!/usr/bin/env python3
"""
RAG Failure Diagnosis Script for UC Berkeley EECS QA System.

Reads a JSONL file of {question, answer, url} and for each question:
  1. Runs hybrid retrieval (top-20 candidates)
  2. Runs reranker (top-5)
  3. Checks if the gold answer appears in each stage
  4. Runs the full RAG pipeline and compares to gold
  5. Categorizes each failure

Usage:
    python3 diagnose_rag.py <qa_dataset.jsonl> [--failures-only] [--limit N]

Output:
    - Per-question diagnosis printed to stdout
    - Summary statistics at the end
"""

import argparse
import json
import os
import re
import sys

# ── bring in your existing RAG system ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag import RAGSystem, FALLBACK_ANSWER

# ---------------------------------------------------------------------------
# Normalisation (mirrors what a fair evaluator would do)
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_url_for_compare(url: str) -> str:
    """Strip scheme, www prefix, and trailing slash for loose comparison."""
    url = url.lower().strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    url = url.rstrip('/')
    return url


def exact_match(pred: str, gold: str) -> bool:
    candidates = [g.strip() for g in gold.split("|")]
    
    if len(candidates) == 1:
        return normalize(pred) == normalize(candidates[0])
    
    # Multi-answer: correct if prediction contains any gold candidate
    pred_lower = normalize(pred)
    return any(normalize(g) in pred_lower for g in candidates)

def f1_score(pred: str, gold: str) -> float:
    candidates = [g.strip() for g in gold.split("|")]
    
    if len(candidates) == 1:
        return _f1_single(pred, candidates[0])
    
    # Multi-answer: check if prediction contains any gold candidate
    pred_lower = normalize(pred)
    for candidate in candidates:
        if normalize(candidate) in pred_lower:
            return 1.0  # predicted at least one correct answer
    
    # Partial credit: best token overlap with any candidate
    return max(_f1_single(pred, g) for g in candidates)

def _f1_single(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Failure categories
# ---------------------------------------------------------------------------

CATEGORY_RETRIEVAL_MISS   = "RETRIEVAL_MISS"      # gold not in top-20
CATEGORY_RERANK_MISS      = "RERANK_MISS"          # gold in top-20 but not top-5
CATEGORY_READER_FAIL      = "READER_FAIL"          # gold in top-5 but wrong answer
CATEGORY_CORRECT          = "CORRECT"
CATEGORY_PARTIAL          = "PARTIAL"              # F1 > 0 but EM wrong


def categorize(in_top20: bool, in_top5: bool, em: bool, f1: float) -> str:
    if em:
        return CATEGORY_CORRECT
    if not in_top20:
        return CATEGORY_RETRIEVAL_MISS
    if not in_top5:
        return CATEGORY_RERANK_MISS
    if f1 == 0.0:
        return CATEGORY_READER_FAIL
    return CATEGORY_PARTIAL


# ---------------------------------------------------------------------------
# Core diagnosis per question
# ---------------------------------------------------------------------------

def diagnose_one(rag: RAGSystem, question: str, gold: str, gold_url: str) -> dict:
    gold_lower = gold.lower().strip()

    # ── Stage 1: hybrid retrieval (top-20) ───────────────────────────────
    candidates = rag.hybrid_retrieve(question)   # returns up to 20 indices
    gold_candidates = [g.strip().lower() for g in gold.split("|")]
    # Check if the gold URL exists anywhere in the corpus at all
    gold_url_normalized = normalize_url_for_compare(gold_url) if gold_url else ""
    existing_urls_normalized = {normalize_url_for_compare(chunk["url"]) for chunk in rag.chunks}
    gold_url_in_corpus = (
        gold_url_normalized in existing_urls_normalized if gold_url else None
    )
    

    in_top20 = any(
    any(g in rag.chunks[idx]["text"].lower() for g in gold_candidates)
    or (gold_url and gold_url.rstrip("/") in rag.chunks[idx]["url"].rstrip("/"))
    for idx in candidates
)
    top20_url_match = any(
        gold_url and gold_url.rstrip("/") in rag.chunks[idx]["url"].rstrip("/")
        for idx in candidates
    ) if gold_url else None

    # ── Stage 2: reranker (top-5) ─────────────────────────────────────────
    top5 = rag.rerank(question, candidates, top_k=5)

    in_top5 = any(
    any(g in rag.chunks[idx]["text"].lower() for g in gold_candidates)
    or (gold_url and gold_url.rstrip("/") in rag.chunks[idx]["url"].rstrip("/"))
    for idx in top5
)
    top5_url_match = any(
        gold_url and gold_url.rstrip("/") in rag.chunks[idx]["url"].rstrip("/")
        for idx in top5
    ) if gold_url else None

    # ── Stage 3: full pipeline answer ────────────────────────────────────
    # Re-use already-retrieved top5 to avoid double LLM calls
    prompt = rag.build_prompt(question, top5)
    try:
        from llm import call_llm
        from rag import LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TIMEOUT, SYSTEM_PROMPT
        raw = call_llm(
            query=prompt,
            system_prompt=SYSTEM_PROMPT,
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            timeout=LLM_TIMEOUT,
        )
        predicted = rag.postprocess(raw)
    except Exception as e:
        predicted = FALLBACK_ANSWER

    em  = exact_match(predicted, gold)
    f1  = f1_score(predicted, gold)
    cat = categorize(in_top20, in_top5, em, f1)

    # ── Best matching chunks for context ─────────────────────────────────
    top5_chunks = [
        {
            "url":  rag.chunks[idx]["url"],
            "text": rag.chunks[idx]["text"][:200],   # truncated for readability
        }
        for idx in top5
    ]

    return {
        "question":      question,
        "gold":          gold,
        "gold_url":      gold_url,
        "gold_url_in_corpus": gold_url_in_corpus,
        "predicted":     predicted,
        "em":            em,
        "f1":            round(f1, 3),
        "in_top20":      in_top20,
        "in_top5":       in_top5,
        "top20_url_hit": top20_url_match,
        "top5_url_hit":  top5_url_match,
        "category":      cat,
        "top5_chunks":   top5_chunks,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    CATEGORY_CORRECT:        "\033[92m",   # green
    CATEGORY_PARTIAL:        "\033[93m",   # yellow
    CATEGORY_READER_FAIL:    "\033[94m",   # blue
    CATEGORY_RERANK_MISS:    "\033[95m",   # magenta
    CATEGORY_RETRIEVAL_MISS: "\033[91m",   # red
}
RESET = "\033[0m"
BOLD  = "\033[1m"


def print_result(r: dict, index: int, verbose: bool = True):
    color = CATEGORY_COLORS.get(r["category"], "")
    print(f"\n{'─'*70}")
    print(f"{BOLD}[{index}] {r['question']}{RESET}")
    print(f"  Gold:      {r['gold']}")
    print(f"  Predicted: {r['predicted']}")
    print(f"  EM: {'✓' if r['em'] else '✗'}   F1: {r['f1']:.3f}")
    print(f"  In top-20: {'✓' if r['in_top20'] else '✗'}   "
          f"In top-5: {'✓' if r['in_top5'] else '✗'}")
    if r["gold_url"]:
        print(f"  Gold URL:           {r['gold_url']}")
        in_corpus = r.get("gold_url_in_corpus")
        corpus_str = "✓ in corpus" if in_corpus else ("✗ NOT IN CORPUS" if in_corpus is False else "unknown")
        print(f"  URL in corpus:      {corpus_str}")
        print(f"  URL in top-20:      {r['top20_url_hit']}   URL in top-5: {r['top5_url_hit']}")
    print(f"  {color}{BOLD}Category: {r['category']}{RESET}")

    if verbose and not r["em"]:
        print(f"\n  {BOLD}Top-5 chunks passed to LLM:{RESET}")
        for i, ch in enumerate(r["top5_chunks"], 1):
            print(f"    [{i}] {ch['url']}")
            print(f"        {ch['text'].strip()[:160]}...")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list):
    total = len(results)
    cats  = {}
    for r in results:
        cats[r["category"]] = cats.get(r["category"], 0) + 1

    avg_f1 = sum(r["f1"] for r in results) / total if total else 0
    em_count = sum(1 for r in results if r["em"])

    print(f"\n{'═'*70}")
    print(f"{BOLD}DIAGNOSIS SUMMARY  ({total} questions){RESET}")
    print(f"{'═'*70}")
    print(f"  Overall EM:  {em_count}/{total}  ({100*em_count/total:.1f}%)")
    print(f"  Overall F1:  {avg_f1:.3f}")
    print()

    order = [
        CATEGORY_CORRECT,
        CATEGORY_PARTIAL,
        CATEGORY_READER_FAIL,
        CATEGORY_RERANK_MISS,
        CATEGORY_RETRIEVAL_MISS,
    ]
    for cat in order:
        n = cats.get(cat, 0)
        pct = 100 * n / total if total else 0
        color = CATEGORY_COLORS.get(cat, "")
        bar = "█" * int(pct / 2)
        print(f"  {color}{cat:<20}{RESET}  {n:3d} ({pct:5.1f}%)  {bar}")

    print()
    print(f"{BOLD}WHAT TO FIX:{RESET}")

    retrieval_miss = cats.get(CATEGORY_RETRIEVAL_MISS, 0)
    rerank_miss    = cats.get(CATEGORY_RERANK_MISS, 0)
    reader_fail    = cats.get(CATEGORY_READER_FAIL, 0)

    total_failures = retrieval_miss + rerank_miss + reader_fail
    if total_failures == 0:
        print("  🎉 No hard failures — focus on partial matches.")
        return

    if retrieval_miss / total_failures > 0.4:
        print("  🔴 RETRIEVAL is your main bottleneck.")
        print("     → Add query expansion / HyDE")
        print("     → Upgrade to bge-base embedding model (requires re-index)")
        print("     → Expand your web crawl — pages may be missing from corpus")
    if rerank_miss / total_failures > 0.3:
        print("  🟣 RERANKER is dropping relevant chunks.")
        print("     → Try cross-encoder/ms-marco-MiniLM-L-12-v2 (stronger)")
        print("     → Increase reranker top_k from 5 to 7")
    if reader_fail / total_failures > 0.3:
        print("  🔵 READER is the bottleneck — chunks are retrieved but answer is wrong.")
        print("     → Improve system prompt")
        print("     → Use a stronger LLM (70B or API model)")
        print("     → Check if answer normalization would fix the mismatch")

    corpus_misses = sum(1 for r in results if r.get("gold_url_in_corpus") is False)
    if corpus_misses > 0:
        print(f"  ⚠️  {corpus_misses} question(s) have a gold URL not present in corpus at all")
        print(f"     → Run inject_url.py on those URLs, then rebuild the index")

    print(f"{'═'*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diagnose RAG failures on a QA JSONL dataset.")
    parser.add_argument("qa_jsonl", help="Path to QA dataset (.jsonl)")
    parser.add_argument("--failures-only", action="store_true",
                        help="Only print questions where EM=0")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only diagnose the first N questions")
    parser.add_argument("--quiet", action="store_true",
                        help="Skip printing top-5 chunk previews")
    args = parser.parse_args()

    # Load dataset
    records = []
    with open(args.qa_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[:args.limit]

    print(f"Loaded {len(records)} questions from {args.qa_jsonl}", file=sys.stderr)

    # Load RAG system
    rag = RAGSystem()

    results = []
    for i, rec in enumerate(records, 1):
        question = rec.get("question", "").strip()
        gold     = rec.get("answer", "").strip()
        gold_url = rec.get("url", "")

        if not question or not gold:
            print(f"  Skipping record {i} (missing question or answer)", file=sys.stderr)
            continue

        print(f"  Diagnosing {i}/{len(records)}: {question[:60]}...", file=sys.stderr)

        result = diagnose_one(rag, question, gold, gold_url)
        results.append(result)

        if args.failures_only and result["em"]:
            continue

        print_result(result, i, verbose=not args.quiet)

    print_summary(results)


if __name__ == "__main__":
    main()