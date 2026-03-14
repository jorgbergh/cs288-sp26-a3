#!/usr/bin/env python3
"""
Generate QA pairs from the crawled EECS corpus using LLM-assisted generation.
Reads corpus.jsonl, produces qa_dataset.jsonl.
"""

import json
import random
import re
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

CORPUS_FILE = "corpus.jsonl"
OUTPUT_FILE = "qa_dataset.jsonl"
TARGET_QA_PAIRS = 150

TOPIC_CATEGORIES = {
    "faculty": ["/people/", "/faculty/", "/Faculty/", "/Homepages/", "/book/faculty"],
    "courses": ["/Courses/", "/courses/", "/academics/courses"],
    "scheduling": ["/Scheduling/", "/schedule"],
    "students": ["/resources/undergrads", "/resources/grads", "/advising/", "/students"],
    "news": ["/news/", "/News/"],
    "awards": ["/awards/", "/award"],
    "phd": ["/phd/", "/doctoral/", "/PhD/", "/graduate"],
    "research": ["/research/", "/Research/", "/labs/"],
    "degree": ["/degree/", "/major/", "/minor/", "/coursework/"],
    "admissions": ["/admissions/", "/Admissions/"],
}

SYSTEM_PROMPT = """You are a QA dataset creator for UC Berkeley's EECS department. Given a webpage's content, generate factoid question-answer pairs.

Rules:
1. Each answer MUST be a short text span (1-10 words) that appears VERBATIM in the provided text.
2. Questions should be specific and have a single unambiguous answer.
3. Vary question types: who, what, when, where, how many, which.
4. Do NOT generate questions about navigation elements, links, or page structure.
5. Focus on factual content: names, numbers, dates, places, titles, degrees. 
6. Output ONLY valid JSON lines, one per question. No other text.

Output format (one JSON per line, no markdown fences):
{"question": "...", "answer": "..."}
{"question": "...", "answer": "..."}"""


def categorize_page(url: str) -> str:
    for category, patterns in TOPIC_CATEGORIES.items():
        if any(p in url for p in patterns):
            return category
    return "other"


def load_corpus():
    docs = []
    with open(CORPUS_FILE) as f:
        for line in f:
            doc = json.loads(line.strip())
            if len(doc["text"]) > 200:
                docs.append(doc)
    return docs


def select_diverse_pages(docs, target_count=75):
    """Select pages with topic diversity."""
    by_category = {}
    for doc in docs:
        cat = categorize_page(doc["url"])
        by_category.setdefault(cat, []).append(doc)

    print(f"Category distribution:", file=sys.stderr)
    for cat, cat_docs in sorted(by_category.items()):
        print(f"  {cat}: {len(cat_docs)} pages", file=sys.stderr)

    selected = []
    categories = list(by_category.keys())

    per_category = max(3, target_count // len(categories))
    for cat in categories:
        cat_docs = by_category[cat]
        cat_docs_sorted = sorted(cat_docs, key=lambda d: len(d["text"]), reverse=True)
        n = min(per_category, len(cat_docs_sorted))
        selected.extend(cat_docs_sorted[:n])

    remaining = target_count - len(selected)
    if remaining > 0:
        all_unselected = [d for d in docs if d not in selected]
        random.shuffle(all_unselected)
        selected.extend(all_unselected[:remaining])

    return selected[:target_count]


def generate_qa_for_page(doc, num_questions=3):
    """Use LLM to generate QA pairs from a single page."""
    text = doc["text"][:4000]
    url = doc["url"]

    prompt = f"""Generate {num_questions} factoid question-answer pairs from this UC Berkeley EECS webpage.

Page URL: {url}
Page title: {doc.get('title', 'N/A')}

Page content:
{text}

Remember: each answer must be a verbatim short span (1-10 words) from the text above. Output ONLY JSON lines."""

    try:
        response = call_llm(
            query=prompt,
            system_prompt=SYSTEM_PROMPT,
            model="meta-llama/llama-3.1-8b-instruct",
            max_tokens=512,
            temperature=0.2,
            timeout=30,
        )
    except Exception as e:
        print(f"  LLM error for {url}: {e}", file=sys.stderr)
        return []

    pairs = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip markdown fences if present
        if line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
            if "question" in obj and "answer" in obj:
                pairs.append(obj)
        except json.JSONDecodeError:
            continue

    return pairs


def validate_answer_in_text(answer: str, text: str) -> bool:
    """Check if the answer appears in the page text (case-insensitive)."""
    return answer.lower() in text.lower()


def main():
    random.seed(42)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: Set OPENROUTER_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    print("Loading corpus...", file=sys.stderr)
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents", file=sys.stderr)

    pages = select_diverse_pages(docs, target_count=75)
    print(f"\nSelected {len(pages)} pages for QA generation", file=sys.stderr)

    all_qa = []
    for i, doc in enumerate(pages):
        num_q = 3 if len(doc["text"]) > 1000 else 2
        pairs = generate_qa_for_page(doc, num_questions=num_q)

        valid_pairs = []
        for pair in pairs:
            if validate_answer_in_text(pair["answer"], doc["text"]):
                pair["url"] = doc["url"]
                valid_pairs.append(pair)
            else:
                print(f"  Dropped (answer not in text): {pair['answer'][:50]}", file=sys.stderr)

        all_qa.extend(valid_pairs)
        print(f"  [{i+1}/{len(pages)}] {doc['url'][:60]}... -> {len(valid_pairs)} valid QA pairs", file=sys.stderr)

        if len(all_qa) >= TARGET_QA_PAIRS:
            print(f"\nReached target of {TARGET_QA_PAIRS} pairs", file=sys.stderr)
            break

        time.sleep(0.5)

    with open(OUTPUT_FILE, "w") as f:
        for qa in all_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"\nGeneration complete!", file=sys.stderr)
    print(f"  Total QA pairs: {len(all_qa)}", file=sys.stderr)
    print(f"  Output: {OUTPUT_FILE}", file=sys.stderr)

    categories = {}
    for qa in all_qa:
        cat = categorize_page(qa["url"])
        categories[cat] = categories.get(cat, 0) + 1
    print(f"  Category breakdown:", file=sys.stderr)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
