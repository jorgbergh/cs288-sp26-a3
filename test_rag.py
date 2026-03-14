#!/usr/bin/env python3
"""
RAG test harness with detailed per-question diagnostics.

Usage:
  python3 test_rag.py                                          # test on reference.jsonl
  python3 test_rag.py --dataset qa_dataset.jsonl               # test on custom dataset
  python3 test_rag.py --question "Where did Dan Klein get his PhD?"  # single question
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter

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

from rag import RAGSystem, SYSTEM_PROMPT, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TIMEOUT, FALLBACK_ANSWER
from llm import call_llm

LINE = "\u2501" * 60


# ---------------------------------------------------------------------------
# Metrics (same as evaluate-v1.1.py)
# ---------------------------------------------------------------------------

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, ground_truth):
    pt = normalize_answer(prediction).split()
    gt = normalize_answer(ground_truth).split()
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    p = ns / len(pt)
    r = ns / len(gt)
    return (2 * p * r) / (p + r)


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def best_metric(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


# ---------------------------------------------------------------------------
# Detailed retrieval
# ---------------------------------------------------------------------------

def retrieve_with_details(rag, question):
    """Run BM25 and dense retrieval separately and tag each chunk's source."""
    bm25_results = rag.retrieve_bm25(question)
    dense_results = rag.retrieve_dense(question)

    bm25_set = {idx for idx, _ in bm25_results}
    dense_set = {idx for idx, _ in dense_results}

    final_indices = rag.hybrid_retrieve(question)

    details = []
    for idx in final_indices:
        sources = []
        if idx in bm25_set:
            sources.append("BM25")
        if idx in dense_set:
            sources.append("Dense")
        chunk = rag.chunks[idx]
        details.append({
            "idx": idx,
            "sources": "+".join(sources) if sources else "RRF",
            "url": chunk["url"],
            "title": chunk.get("title", ""),
            "text": chunk["text"],
        })
    return final_indices, details


# ---------------------------------------------------------------------------
# Single question mode
# ---------------------------------------------------------------------------

def test_single_question(rag, question):
    print(f"\n{LINE}")
    print(f"Question: {question}")
    print(LINE)

    chunk_indices, details = retrieve_with_details(rag, question)

    print(f"\nRetrieved chunks (top {len(details)}):")
    for i, d in enumerate(details, 1):
        snippet = d["text"][:120].replace("\n", " ")
        print(f"  [{i}] {d['sources']:12s} | {d['url']}")
        print(f"      \"{snippet}...\"")

    prompt = rag.build_prompt(question, chunk_indices)

    print(f"\nPrompt ({len(prompt)} chars):")
    print(f"  System: {SYSTEM_PROMPT[:80]}...")
    print(f"  User:   {prompt[:120]}...")

    t0 = time.time()
    try:
        raw_answer = call_llm(
            query=prompt,
            system_prompt=SYSTEM_PROMPT,
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            timeout=LLM_TIMEOUT,
        )
    except Exception as e:
        raw_answer = f"[ERROR: {e}]"
    elapsed = time.time() - t0

    processed = rag.postprocess(raw_answer) if not raw_answer.startswith("[ERROR") else FALLBACK_ANSWER

    print(f"\nLLM response ({elapsed:.2f}s): \"{raw_answer}\"")
    print(f"Post-processed:           \"{processed}\"")
    print(LINE)


# ---------------------------------------------------------------------------
# Dataset evaluation mode
# ---------------------------------------------------------------------------

def test_dataset(rag, dataset_path):
    with open(dataset_path) as f:
        qa_pairs = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"\nEvaluating {len(qa_pairs)} questions from {dataset_path}\n")

    results = []
    total_em = total_f1 = 0.0
    url_hits = answer_hits = 0
    retrieval_misses = generation_misses = 0

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        gold_answers = [a.strip() for a in qa["answer"].split("|")]
        gold_url = qa.get("url", "")

        print(f"{LINE}")
        print(f"Q[{i+1}/{len(qa_pairs)}]: {question}")
        print(f"Gold:    {' | '.join(gold_answers)}")
        if gold_url:
            print(f"URL:     {gold_url}")
        print(LINE)

        chunk_indices, details = retrieve_with_details(rag, question)

        print(f"\nRetrieved chunks (top {len(details)}):")
        for j, d in enumerate(details, 1):
            snippet = d["text"][:120].replace("\n", " ")
            print(f"  [{j}] {d['sources']:12s} | {d['url']}")
            print(f"      \"{snippet}...\"")

        # Retrieval diagnostics
        url_found = False
        url_rank = None
        answer_found = False
        answer_rank = None

        for rank, d in enumerate(details, 1):
            if gold_url and d["url"] == gold_url:
                if not url_found:
                    url_found = True
                    url_rank = rank
            for ga in gold_answers:
                if ga.lower() in d["text"].lower():
                    if not answer_found:
                        answer_found = True
                        answer_rank = rank
                    break

        if url_found:
            url_hits += 1
        if answer_found:
            answer_hits += 1

        url_str = f"YES (rank {url_rank})" if url_found else "NO"
        ans_str = f"YES (chunk {answer_rank})" if answer_found else "NO"
        print(f"\nRetrieval diagnostics:")
        print(f"  Gold URL found:    {url_str}")
        print(f"  Answer in chunks:  {ans_str}")

        # Generate answer
        prompt = rag.build_prompt(question, chunk_indices)
        t0 = time.time()
        try:
            raw_answer = call_llm(
                query=prompt,
                system_prompt=SYSTEM_PROMPT,
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                timeout=LLM_TIMEOUT,
            )
        except Exception as e:
            raw_answer = f"[ERROR: {e}]"
        elapsed = time.time() - t0

        processed = rag.postprocess(raw_answer) if not raw_answer.startswith("[ERROR") else FALLBACK_ANSWER

        em = best_metric(exact_match_score, processed, gold_answers)
        f1 = best_metric(f1_score, processed, gold_answers)
        total_em += em
        total_f1 += f1

        # Classify error type
        error_type = None
        if f1 == 0:
            if not answer_found:
                error_type = "retrieval_miss"
                retrieval_misses += 1
            else:
                error_type = "generation_miss"
                generation_misses += 1

        print(f"\nLLM response ({elapsed:.2f}s): \"{raw_answer}\"")
        print(f"Post-processed:           \"{processed}\"")
        print(f"\nMetrics:  EM={em:.0f}  F1={f1:.2f}", end="")
        if error_type:
            print(f"  [{error_type.upper()}]", end="")
        print()

        results.append({
            "question": question,
            "gold_answers": gold_answers,
            "gold_url": gold_url,
            "prediction": processed,
            "raw_llm_response": raw_answer,
            "em": em,
            "f1": f1,
            "url_retrieved": url_found,
            "answer_in_chunks": answer_found,
            "error_type": error_type,
            "retrieved_urls": [d["url"] for d in details],
            "llm_latency_s": elapsed,
        })

    # Aggregate summary
    n = len(qa_pairs)
    avg_em = total_em / n
    avg_f1 = total_f1 / n
    url_recall = url_hits / n
    ans_recall = answer_hits / n
    errors = sum(1 for r in results if r["f1"] == 0)

    print(f"\n{'=' * 60}")
    print(f"AGGREGATE RESULTS — {dataset_path}")
    print(f"{'=' * 60}")
    print(f"  Questions:          {n}")
    print(f"  Exact Match:        {avg_em:.3f} ({avg_em*100:.1f}%)")
    print(f"  F1 Score:           {avg_f1:.3f} ({avg_f1*100:.1f}%)")
    print(f"  Retrieval URL:      {url_recall:.3f} ({url_recall*100:.1f}%)")
    print(f"  Retrieval Answer:   {ans_recall:.3f} ({ans_recall*100:.1f}%)")
    print(f"  ───────────────────────────────────────")
    print(f"  Total errors (F1=0): {errors}")
    print(f"    Retrieval misses:  {retrieval_misses} (answer not in any retrieved chunk)")
    print(f"    Generation misses: {generation_misses} (answer in chunks but LLM failed)")
    print(f"{'=' * 60}")

    # Save results
    output = {
        "dataset": dataset_path,
        "summary": {
            "n": n,
            "exact_match": avg_em,
            "f1": avg_f1,
            "retrieval_url_recall": url_recall,
            "retrieval_answer_recall": ans_recall,
            "total_errors": errors,
            "retrieval_misses": retrieval_misses,
            "generation_misses": generation_misses,
        },
        "per_question": results,
    }
    with open("test_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nDetailed results saved to test_results.json")

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG test harness")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to JSONL evaluation dataset (default: reference.jsonl)")
    parser.add_argument("--question", type=str, default=None,
                        help="Test a single question interactively")
    args = parser.parse_args()

    rag = RAGSystem()

    if args.question:
        test_single_question(rag, args.question)
    else:
        dataset = args.dataset or "reference.jsonl"
        if not os.path.exists(dataset):
            print(f"Error: dataset file '{dataset}' not found", file=sys.stderr)
            sys.exit(1)
        test_dataset(rag, dataset)


if __name__ == "__main__":
    main()
