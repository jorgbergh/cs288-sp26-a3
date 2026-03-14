#!/usr/bin/env python3
"""
Measure Inter-Annotator Agreement (IAA) on the QA dataset.
Uses two different LLMs as independent annotators on a 35% subset.
"""

import json
import random
import re
import string
import sys
import time
import os

def load_dotenv(path=".env"):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass

load_dotenv()

from llm import call_llm

QA_FILE = "qa_dataset.jsonl"
CORPUS_FILE = "corpus.jsonl"
IAA_FRACTION = 0.35

MODEL_A = "meta-llama/llama-3.1-8b-instruct"
MODEL_B = "qwen/qwen-2.5-7b-instruct"


def normalize_answer(s):
    """Normalize answer for comparison (same as evaluate-v1.1.py)."""
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
    from collections import Counter
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def cohens_kappa(labels_a, labels_b):
    """Compute Cohen's Kappa for two binary label lists."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 0.0

    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n

    pos_a = sum(labels_a) / n
    pos_b = sum(labels_b) / n
    p_e = pos_a * pos_b + (1 - pos_a) * (1 - pos_b)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def load_corpus_by_url():
    url_to_text = {}
    with open(CORPUS_FILE) as f:
        for line in f:
            doc = json.loads(line.strip())
            url_to_text[doc["url"]] = doc["text"]
    return url_to_text


def answer_question(question, context, model):
    """Have a model answer a question given context."""
    system = "Answer the question using ONLY the provided context. Give a short answer (1-10 words). Output ONLY the answer, nothing else."
    prompt = f"Context:\n{context[:3000]}\n\nQuestion: {question}\n\nAnswer:"

    try:
        return call_llm(
            query=prompt,
            system_prompt=system,
            model=model,
            max_tokens=32,
            temperature=0.0,
            timeout=30,
        )
    except Exception as e:
        print(f"  Error ({model}): {e}", file=sys.stderr)
        return ""


def main():
    random.seed(42)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    print("Loading QA dataset...", file=sys.stderr)
    qa_pairs = []
    with open(QA_FILE) as f:
        for line in f:
            qa_pairs.append(json.loads(line.strip()))
    print(f"  Total QA pairs: {len(qa_pairs)}", file=sys.stderr)

    print("Loading corpus...", file=sys.stderr)
    url_to_text = load_corpus_by_url()

    subset_size = max(30, int(len(qa_pairs) * IAA_FRACTION))
    subset = random.sample(qa_pairs, min(subset_size, len(qa_pairs)))
    print(f"  IAA subset size: {len(subset)} ({100*len(subset)/len(qa_pairs):.0f}%)", file=sys.stderr)

    results = []
    for i, qa in enumerate(subset):
        question = qa["question"]
        gold_answer = qa["answer"]
        url = qa["url"]

        context = url_to_text.get(url, "")
        if not context:
            print(f"  Skipping (no context): {question[:50]}", file=sys.stderr)
            continue

        answer_a = answer_question(question, context, MODEL_A)
        time.sleep(0.3)
        answer_b = answer_question(question, context, MODEL_B)
        time.sleep(0.3)

        gold_answers = [g.strip() for g in gold_answer.split("|")]

        em_a = max(exact_match(answer_a, g) for g in gold_answers)
        em_b = max(exact_match(answer_b, g) for g in gold_answers)
        f1_a = max(f1_score(answer_a, g) for g in gold_answers)
        f1_b = max(f1_score(answer_b, g) for g in gold_answers)

        em_ab = exact_match(answer_a, answer_b)
        f1_ab = f1_score(answer_a, answer_b)

        results.append({
            "question": question,
            "gold": gold_answer,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "em_a": em_a,
            "em_b": em_b,
            "f1_a": f1_a,
            "f1_b": f1_b,
            "em_ab": em_ab,
            "f1_ab": f1_ab,
        })

        print(f"  [{i+1}/{len(subset)}] Q: {question[:50]}...", file=sys.stderr)
        print(f"    Gold: {gold_answer} | A: {answer_a} | B: {answer_b}", file=sys.stderr)

    if not results:
        print("No results to compute IAA", file=sys.stderr)
        return

    avg_em_a = sum(r["em_a"] for r in results) / len(results)
    avg_em_b = sum(r["em_b"] for r in results) / len(results)
    avg_f1_a = sum(r["f1_a"] for r in results) / len(results)
    avg_f1_b = sum(r["f1_b"] for r in results) / len(results)

    avg_em_ab = sum(r["em_ab"] for r in results) / len(results)
    avg_f1_ab = sum(r["f1_ab"] for r in results) / len(results)

    labels_a = [int(r["em_a"] > 0) for r in results]
    labels_b = [int(r["em_b"] > 0) for r in results]
    kappa = cohens_kappa(labels_a, labels_b)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"INTER-ANNOTATOR AGREEMENT RESULTS", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Subset size: {len(results)} questions", file=sys.stderr)
    print(f"Annotator A: {MODEL_A}", file=sys.stderr)
    print(f"Annotator B: {MODEL_B}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"Annotator A vs Gold:", file=sys.stderr)
    print(f"  Exact Match: {avg_em_a:.3f}", file=sys.stderr)
    print(f"  F1 Score:    {avg_f1_a:.3f}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"Annotator B vs Gold:", file=sys.stderr)
    print(f"  Exact Match: {avg_em_b:.3f}", file=sys.stderr)
    print(f"  F1 Score:    {avg_f1_b:.3f}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"Annotator A vs Annotator B:", file=sys.stderr)
    print(f"  Exact Match: {avg_em_ab:.3f}", file=sys.stderr)
    print(f"  F1 Score:    {avg_f1_ab:.3f}", file=sys.stderr)
    print(f"  Cohen's Kappa (on EM correctness): {kappa:.3f}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    with open("iaa_results.json", "w") as f:
        json.dump({
            "subset_size": len(results),
            "model_a": MODEL_A,
            "model_b": MODEL_B,
            "annotator_a_em": avg_em_a,
            "annotator_a_f1": avg_f1_a,
            "annotator_b_em": avg_em_b,
            "annotator_b_f1": avg_f1_b,
            "ab_em": avg_em_ab,
            "ab_f1": avg_f1_ab,
            "cohens_kappa": kappa,
            "details": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to iaa_results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
