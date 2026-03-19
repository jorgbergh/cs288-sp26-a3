#!/usr/bin/env python3
"""
Manually inject a URL into corpus.jsonl using the same extraction
and normalization logic as crawl_eecs.py.

Usage:
    python inject_url.py <url> [<url2> ...]
    python inject_url.py https://eecs.berkeley.edu/some-page
"""

import json
import re
import sys
from urllib.parse import urldefrag, urlparse

import requests
from bs4 import BeautifulSoup
import io
import pdfplumber

OUTPUT_FILE = "corpus.jsonl"
TIMEOUT = 8


def extract_pdf_text(content: bytes) -> tuple:
    """Extract title and text from raw PDF bytes using pdfplumber."""
    title = ""
    pages_text = []
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            # Try to get title from metadata
            meta = pdf.metadata or {}
            title = meta.get("Title", "").strip()

            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text.strip())
    except Exception as e:
        print(f"[warn] PDF parse error: {e}")

    full_text = "\n\n".join(pages_text)
    full_text = re.sub(r' +', ' ', full_text)
    full_text = re.sub(r'\n\s*\n+', '\n\n', full_text)
    return title, full_text.strip()

def normalize_url(url: str) -> str:
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def extract_table_text(table) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            cells.append(td.get_text(separator=" ", strip=True))
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_text(html: str) -> tuple:
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string.strip() if soup.title else ""

    for tag in soup.find_all(["script", "style", "noscript", "iframe", "nav", "footer", "header"]):
        tag.decompose()

    tables_text = []
    for table in soup.find_all("table"):
        tables_text.append(extract_table_text(table))
        table.decompose()

    content_container = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"id": "content"})
        or soup.find("div", {"class": re.compile(r"content|main|body", re.I)})
    )

    target = content_container if content_container else soup.body
    if not target:
        target = soup

    text = target.get_text(separator="\n", strip=True)

    if tables_text:
        text = text + "\n\n" + "\n\n".join(tables_text)

    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return title, text.strip()


def load_existing_urls(output_file: str) -> set:
    urls = set()
    try:
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    raw = doc.get("url", "")
                    urls.add(raw)
                    urls.add(normalize_url(raw))  # add normalized form too
    except FileNotFoundError:
        pass
    return urls


def inject_url(url: str, output_file: str = OUTPUT_FILE, skip_duplicates: bool = True) -> bool:
    """
    Fetch a URL, extract its text, and append it to corpus.jsonl.

    Args:
        url:             The URL to fetch and inject.
        output_file:     Path to the corpus JSONL file.
        skip_duplicates: If True, skip URLs already in the corpus.

    Returns:
        True if a new entry was written, False otherwise.
    """
    norm = normalize_url(url)

    if skip_duplicates:
        existing = load_existing_urls(output_file)
        if norm in existing:
            print(f"[skip] Already in corpus: {norm}")
            return False

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Educational Bot for RAG Research)"})

    try:
        resp = session.get(norm, timeout=(5, TIMEOUT), allow_redirects=True)
    except Exception as e:
        print(f"[error] Could not fetch {norm}: {e}")
        return False

    if resp.status_code != 200:
        print(f"[error] HTTP {resp.status_code} for {norm}")
        return False

    content_type = resp.headers.get("Content-Type", "").lower()

    # Use the final URL after redirects
    final_url = normalize_url(resp.url)

    if skip_duplicates and final_url != norm:
        existing = load_existing_urls(output_file)
        if final_url in existing:
            print(f"[skip] Redirected URL already in corpus: {final_url}")
            return False

    if "application/pdf" in content_type or final_url.lower().endswith(".pdf"):
        title, text = extract_pdf_text(resp.content)
    elif "text/html" in content_type:
        title, text = extract_text(resp.text)
    else:
        print(f"[error] Unsupported Content-Type ({content_type}): {final_url}")
        return False

    # Use URL as fallback title for PDFs with no metadata title
    if not title:
        title = final_url.split("/")[-1].replace(".pdf", "").replace("-", " ").replace("_", " ")

    if len(text) < 200:
        print(f"[warn] Page has very little content ({len(text)} chars), injecting anyway: {final_url}")

    doc = {"url": final_url, "title": title, "text": text}

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"[ok] Injected: {final_url}  (title: {title!r}, {len(text)} chars)")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <url> [<url2> ...]")
        sys.exit(1)

    urls = sys.argv[1:]
    added = 0
    for url in urls:
        if inject_url(url):
            added += 1

    print(f"\nDone. {added}/{len(urls)} URL(s) added to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
