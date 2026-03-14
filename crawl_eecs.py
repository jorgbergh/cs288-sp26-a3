#!/usr/bin/env python3
"""
BFS crawler for eecs.berkeley.edu.
Outputs corpus.jsonl with one JSON object per page: {url, title, text}.
Writes incrementally so data is preserved even if interrupted.
"""

import json
import re
import time
import sys
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

SEED_URLS = [
    "https://eecs.berkeley.edu/",
    "https://eecs.berkeley.edu/people/",
    "https://eecs.berkeley.edu/resources/",
    "https://eecs.berkeley.edu/research/",
    "https://eecs.berkeley.edu/academics/",
    "https://eecs.berkeley.edu/news/",
    "https://www2.eecs.berkeley.edu/",
    "https://www2.eecs.berkeley.edu/Courses/CS/",
    "https://www2.eecs.berkeley.edu/Courses/EE/",
    "https://www2.eecs.berkeley.edu/Faculty/Homepages/",
    "https://www2.eecs.berkeley.edu/Scheduling/CS/schedule.html",
    "https://www2.eecs.berkeley.edu/Scheduling/EE/schedule.html",
]

URL_PATTERN = re.compile(
    r"https?://(?:www\d*\.)?eecs\.berkeley\.edu(?:/[^\s]*)?"
)

SKIP_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico",
    ".zip", ".tar", ".gz", ".bz2",
    ".pptx", ".ppt", ".doc", ".docx", ".xls", ".xlsx",
    ".mp4", ".mp3", ".avi", ".mov", ".wmv",
    ".css", ".js", ".json", ".xml", ".rss",
    ".bib", ".ps", ".eps", ".dvi", ".tex",
}

SKIP_PATH_PATTERNS = [
    "/cgi-bin/",
    "/~",          # personal homepages tend to be very old/broken
    "/Pubs/TechRpts/19",  # very old tech reports from the 1900s
]

MAX_PAGES = 2000
DELAY = 0.25
TIMEOUT = 8
OUTPUT_FILE = "corpus.jsonl"


def normalize_url(url: str) -> str:
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def should_skip_url(url: str) -> bool:
    parsed = urlparse(url)
    path_lower = parsed.path.lower()

    if any(path_lower.endswith(ext) for ext in SKIP_EXTENSIONS):
        return True
    if "login" in path_lower or "signin" in path_lower:
        return True
    if parsed.scheme not in ("http", "https"):
        return True
    if not URL_PATTERN.match(url):
        return True
    for pattern in SKIP_PATH_PATTERNS:
        if pattern in parsed.path:
            return True
    return False


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

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    for tag in soup.find_all(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    tables_text = []
    for table in soup.find_all("table"):
        tables_text.append(extract_table_text(table))
        table.decompose()

    content_container = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"id": "content"})
        or soup.find("div", {"class": re.compile(r"content|main", re.I)})
        or soup.body
        or soup
    )

    text = content_container.get_text(separator="\n", strip=True)

    if tables_text:
        text = text + "\n\n" + "\n\n".join(tables_text)

    lines = text.split("\n")
    cleaned = [line.strip() for line in lines if line.strip()]
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return title, text


def crawl():
    visited = set()
    queue = deque()

    for url in SEED_URLS:
        norm = normalize_url(url)
        if norm not in visited:
            visited.add(norm)
            queue.append(norm)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (UC Berkeley CS288 Assignment Crawler)"
    })

    page_count = 0
    errors = 0
    consecutive_errors = 0

    with open(OUTPUT_FILE, "w") as f:
        while queue and page_count < MAX_PAGES:
            url = queue.popleft()

            try:
                resp = session.get(url, timeout=TIMEOUT, allow_redirects=True)
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    continue
                resp.raise_for_status()
                consecutive_errors = 0
            except Exception:
                errors += 1
                consecutive_errors += 1
                if errors % 100 == 0:
                    print(f"  [{errors} errors so far, queue: {len(queue)}]", file=sys.stderr)
                continue

            page_count += 1
            if page_count % 50 == 0:
                print(f"  Crawled {page_count} pages, queue size: {len(queue)}, errors: {errors}", file=sys.stderr)

            title, text = extract_text(resp.text)

            if len(text.strip()) < 50:
                continue

            doc = {"url": url, "title": title, "text": text}
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            f.flush()

            soup = BeautifulSoup(resp.text, "lxml")
            for link_tag in soup.find_all("a", href=True):
                href = link_tag["href"]
                absolute = urljoin(url, href)
                norm = normalize_url(absolute)

                if norm in visited:
                    continue
                if should_skip_url(norm):
                    continue

                visited.add(norm)
                queue.append(norm)

            time.sleep(DELAY)

    return page_count, errors


def main():
    print(f"Starting EECS website crawl...", file=sys.stderr)
    print(f"Seed URLs: {len(SEED_URLS)}", file=sys.stderr)
    print(f"Max pages: {MAX_PAGES}", file=sys.stderr)

    page_count, errors = crawl()

    print(f"\nCrawl complete!", file=sys.stderr)
    print(f"  Pages crawled: {page_count}", file=sys.stderr)
    print(f"  Errors: {errors}", file=sys.stderr)
    print(f"  Output: {OUTPUT_FILE}", file=sys.stderr)

    line_count = 0
    total_chars = 0
    with open(OUTPUT_FILE) as f:
        for line in f:
            doc = json.loads(line)
            total_chars += len(doc["text"])
            line_count += 1
    print(f"  Documents saved: {line_count}", file=sys.stderr)
    print(f"  Total text: {total_chars:,} characters", file=sys.stderr)


if __name__ == "__main__":
    main()
