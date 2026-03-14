#!/usr/bin/env python3
"""
Local evaluation script for the RAG system.
Tests on reference.jsonl and qa_dataset.jsonl, reports EM, F1, and retrieval recall.
"""

import json
import os
import re
import string
import sys
import tempfile
from collections import Counter

# Load .env if present
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from rag import RAGSystem


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def evaluate_dataset(dataset_path, rag):
    """Evaluate RAG on a JSONL dataset."""
    with open(dataset_path) as f:
        qa_pairs = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"\nEvaluating on {dataset_path} ({len(qa_pairs)} questions)...", file=sys.stderr)

    total_em = 0.0
    total_f1 = 0.0
    total = 0
    retrieval_hits = 0
    errors = []

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        gold_answers = [a.strip() for a in qa["answer"].split("|")]
        gold_url = qa.get("url", "")

        prediction = rag.answer_question(question)

        em = metric_max_over_ground_truths(exact_match_score, prediction, gold_answers)
        f1 = metric_max_over_ground_truths(f1_score, prediction, gold_answers)

        total_em += em
        total_f1 += f1
        total += 1

        # Check retrieval recall
        chunk_indices = rag.hybrid_retrieve(question)
        hit = False
        for idx in chunk_indices:
            chunk = rag.chunks[idx]
            chunk_text_lower = chunk["text"].lower()
            chunk_url = chunk["url"]
            if gold_url and chunk_url == gold_url:
                hit = True
                break
            for ga in gold_answers:
                if ga.lower() in chunk_text_lower:
                    hit = True
                    break
            if hit:
                break
        if hit:
            retrieval_hits += 1

        status = "OK" if f1 > 0 else "MISS"
        print(f"  [{i+1}/{len(qa_pairs)}] {status} | F1={f1:.2f} EM={em:.0f} | "
              f"Q: {question[:50]}... | Pred: {prediction[:40]} | Gold: {gold_answers[0][:40]}",
              file=sys.stderr)

        if f1 == 0:
            errors.append({
                "question": question,
                "prediction": prediction,
                "gold": gold_answers,
                "url": gold_url,
            })

    avg_em = total_em / total if total > 0 else 0
    avg_f1 = total_f1 / total if total > 0 else 0
    retrieval_recall = retrieval_hits / total if total > 0 else 0

    return {
        "dataset": dataset_path,
        "total": total,
        "exact_match": avg_em,
        "f1": avg_f1,
        "retrieval_recall": retrieval_recall,
        "errors": errors,
    }


def main():
    rag = RAGSystem()

    datasets = []
    if os.path.exists("reference.jsonl"):
        datasets.append("reference.jsonl")
    if os.path.exists("qa_dataset.jsonl"):
        datasets.append("qa_dataset.jsonl")

    if not datasets:
        print("No evaluation datasets found", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for ds in datasets:
        result = evaluate_dataset(ds, rag)
        all_results.append(result)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"EVALUATION RESULTS", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for r in all_results:
        print(f"\n{r['dataset']}:", file=sys.stderr)
        print(f"  Total questions: {r['total']}", file=sys.stderr)
        print(f"  Exact Match:     {r['exact_match']:.3f} ({r['exact_match']*100:.1f}%)", file=sys.stderr)
        print(f"  F1 Score:        {r['f1']:.3f} ({r['f1']*100:.1f}%)", file=sys.stderr)
        print(f"  Retrieval Recall:{r['retrieval_recall']:.3f} ({r['retrieval_recall']*100:.1f}%)", file=sys.stderr)
        if r['errors']:
            print(f"  Errors (F1=0):   {len(r['errors'])}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Save detailed results
    with open("eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to eval_results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
