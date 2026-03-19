#!/usr/bin/env python3
"""
BFS crawler for eecs.berkeley.edu.
Outputs corpus.jsonl with one JSON object per page: {url, title, text}.
Writes incrementally so data is preserved even if interrupted.

Strategy: priority queue ensures high-value pages (courses, faculty, academics,
research) are crawled first. Per-category caps prevent news/blog flooding.
"""

import json
import re
import time
import sys
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
import random

import requests
from bs4 import BeautifulSoup

# Seed URLs — covers every major section of the EECS site.
# High-value structural pages are listed first so they seed the priority queue.
SEED_URLS = [
    # Core department info
    "https://eecs.berkeley.edu/",
    "https://eecs.berkeley.edu/about/",
    "https://eecs.berkeley.edu/contact/",

    # Academics — most likely to be tested
    "https://eecs.berkeley.edu/academics/",
    "https://eecs.berkeley.edu/academics/undergrad/",
    "https://eecs.berkeley.edu/academics/undergrad/eecs/",
    "https://eecs.berkeley.edu/academics/undergrad/cs/",
    "https://eecs.berkeley.edu/academics/undergrad/eecs/",
    "https://eecs.berkeley.edu/academics/grad/",
    "https://eecs.berkeley.edu/academics/grad/masters/",
    "https://eecs.berkeley.edu/academics/grad/phd/",

    # Courses (www2) — individual course pages contain the richest structured data
    "https://www2.eecs.berkeley.edu/Courses/CS/",
    "https://www2.eecs.berkeley.edu/Courses/EE/",

    # Scheduling
    "https://www2.eecs.berkeley.edu/Scheduling/CS/schedule.html",
    "https://www2.eecs.berkeley.edu/Scheduling/EE/schedule.html",

    # Faculty
    "https://eecs.berkeley.edu/people/faculty/",
    "https://www2.eecs.berkeley.edu/Faculty/Homepages/",

    # Research areas
    "https://eecs.berkeley.edu/research/",
    "https://www2.eecs.berkeley.edu/Research/Areas/",

    # People / community
    "https://eecs.berkeley.edu/people/",
    "https://eecs.berkeley.edu/people/staff/",
    "https://eecs.berkeley.edu/people/grad-students/",

    # Resources
    "https://eecs.berkeley.edu/resources/",
    "https://eecs.berkeley.edu/resources/students/",
    "https://eecs.berkeley.edu/resources/grads/",

    # Industry / connect
    "https://eecs.berkeley.edu/industry/",
    "https://eecs.berkeley.edu/connect/",

    # www2 root (has links to many subsections)
    "https://www2.eecs.berkeley.edu/",

    # A few recent news entries are fine — cap enforced below
    "https://eecs.berkeley.edu/news/",
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

# Paths to skip entirely — low-value or duplicate content
SKIP_PATH_PATTERNS = [
    "/cgi-bin/",
    "/~",               # personal homepages — old/broken
    "/Pubs/TechRpts/19",  # pre-2000 tech reports
    "/category/",       # WordPress tag/category archive pages
    "/tag/",            # WordPress tag pages
    "/author/",         # WordPress author pages
    "/page/",           # WordPress pagination (/page/2/, /page/3/, …)
    "/feed/",           # RSS/Atom feeds
    "/wp-json/",        # WordPress REST API
    "/wp-admin/",
    "/wp-content/",
    "/wp-includes/",
    "/trackback/",
    "/xmlrpc",
    "/2013/",          # very old dated archives
    "/2012/",
    "/2011/",
    "/2010/",
    "/2009/",
    "/2008/",
    "/2007/",
    "/2006/",
    "/2005/",
    "/2004/"
]

# Paths that get crawled from the priority queue (front of the BFS).
# Anything matching these prefixes is important structural content.
PRIORITY_PATH_PREFIXES = (
    "/academics",
    "/Courses",
    "/Scheduling",
    "/Faculty",
    "/Research",
    "/research",
    "/people",
    "/resources",
    "/about",
    "/industry",
    "/connect",
    "/grad",
    "/undergrad",
)

# Per-path-prefix crawl caps — prevents any one section from flooding the corpus.
# Keys are lowercase path prefixes; values are max documents to save.
PATH_CAPS = {
    "/news":  150,
    "/book":   30,
    "/pubs":   60,
    "/blog":   30,
    "/media":  20,
    "/events": 30,
}

MAX_PAGES = 2000
DELAY = 1.0
TIMEOUT = 11          # read timeout (seconds) — increased from 8
CONNECT_TIMEOUT = 8   # connection timeout — increased from 5
PRIORITY_TIMEOUT = 25 # longer timeout for high-value pages
MAX_RETRIES = 3       # how many times to retry a failed URL
RETRY_BACKOFF = 2.0   # seconds to wait before first retry (doubles each attempt)
RANDOM_SEED_COUNT = 100
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


def is_priority_url(url: str) -> bool:
    """Return True if this URL belongs to a high-value structural section."""
    path = urlparse(url).path
    return any(path.startswith(p) for p in PRIORITY_PATH_PREFIXES)


def get_cap_key(url: str):
    """Return the PATH_CAPS key that applies to this URL, or None."""
    path = urlparse(url).path.lower()
    for prefix in PATH_CAPS:
        if path.startswith(prefix) or ("/" + prefix.lstrip("/")) in path:
            return prefix
    return None


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

    # 1. Remove non-content elements
    for tag in soup.find_all(["script", "style", "noscript", "iframe", "nav", "footer", "header"]):
        tag.decompose()

    # 2. Extract tables before decomposing them
    tables_text = []
    for table in soup.find_all("table"):
        tables_text.append(extract_table_text(table))
        table.decompose()

    # 3. Targeted content extraction — prefer semantic containers
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

    # 4. Whitespace cleanup
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return title, text.strip()


def crawl():
    # written_urls: URLs already saved to corpus.jsonl — never re-write these.
    # visited: URLs already queued or processed this session — never re-queue these.
    # Keeping them separate lets us re-queue seed URLs (to rediscover outgoing
    # links) even when their content is already in the corpus.
    written_urls = set()
    visited = set()
    path_counts = {k: 0 for k in PATH_CAPS}
    existing_count = 0
    page_count = 0  # counts only NEW pages written this session

    # Resume support: pre-load already-written URLs so we skip writing them again.
    import os
    if os.path.exists(OUTPUT_FILE):
        print(f"Resuming — loading existing corpus from {OUTPUT_FILE}...", file=sys.stderr)
        with open(OUTPUT_FILE, encoding="utf-8") as existing:
            for line in existing:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    norm = normalize_url(doc["url"])
                    written_urls.add(norm)
                    cap_key = get_cap_key(norm)
                    if cap_key is not None:
                        path_counts[cap_key] += 1
                    existing_count += 1
                except Exception:
                    pass
        print(f"  Loaded {existing_count} already-crawled pages (will not re-write).", file=sys.stderr)
    if written_urls:
        sample_size = min(RANDOM_SEED_COUNT, len(written_urls))
        random_seeds = random.sample(sorted(written_urls), sample_size)
        print(f"  Adding {sample_size} randomly sampled URLs to seed queue for link rediscovery.", file=sys.stderr)
        SEED_URLS.extend(random_seeds)

    # Two queues: priority items go to the front (appendleft), others to the back.
    # Seed URLs are always re-queued so their outgoing links get rediscovered,
    # even if those pages are already in the corpus.
    priority_queue = deque()
    normal_queue = deque()

    for url in SEED_URLS:
        norm = normalize_url(url)
        if norm not in visited:
            visited.add(norm)
            if is_priority_url(norm):
                priority_queue.append(norm)
            else:
                normal_queue.append(norm)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Educational Bot for RAG Research)"
    })

    fetch_count = 0
    errors = 0
    skip_already_written = 0
    skip_short_text = 0
    skip_cap = 0

    def next_url():
        if priority_queue:
            return priority_queue.popleft()
        return normal_queue.popleft()

    def has_urls():
        return priority_queue or normal_queue
    
    def get_timeout(url: str):
        if "www2" in url:
            return (10, 40)  # longer connect and read timeout
        return (CONNECT_TIMEOUT, TIMEOUT)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        while has_urls() and page_count < MAX_PAGES:
            url = next_url()
            priority = is_priority_url(url)
            read_timeout = PRIORITY_TIMEOUT if priority else TIMEOUT

            resp = None
            last_exc = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    resp = session.get(
                        url,
                        timeout=get_timeout(url),
                        allow_redirects=True,
                    )
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    fetch_count += 1
                    if attempt < MAX_RETRIES:
                        wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                        print(
                            f"  Retry {attempt}/{MAX_RETRIES - 1} for {url} "
                            f"(wait {wait:.1f}s): {e}",
                            file=sys.stderr,
                        )
                        time.sleep(wait)

            fetch_count += 1  # count the successful fetch (or final failure)

            if last_exc is not None:
                errors += 1
                print(f"  Failed after {MAX_RETRIES} attempts: {url}", file=sys.stderr)
                if fetch_count % 20 == 0:
                    print(
                        f"Progress: {page_count}/{MAX_PAGES} saved | "
                        f"fetched={fetch_count} errors={errors} "
                        f"pq={len(priority_queue)} nq={len(normal_queue)} | "
                        f"skip: written={skip_already_written} short={skip_short_text} cap={skip_cap}",
                        file=sys.stderr,
                    )
                continue

            if resp.status_code != 200:
                continue

            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type:
                continue

            final_url = normalize_url(resp.url)
            if final_url != url and final_url in visited:
                continue
            visited.add(final_url)

            title, text = extract_text(resp.text)

            if final_url in written_urls:
                skip_already_written += 1
            elif len(text) < 200:
                skip_short_text += 1
            else:
                cap_key = get_cap_key(final_url)
                if cap_key is not None and path_counts[cap_key] >= PATH_CAPS[cap_key]:
                    skip_cap += 1
                else:
                    doc = {"url": final_url, "title": title, "text": text}
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    f.flush()
                    written_urls.add(final_url)
                    page_count += 1
                    if cap_key is not None:
                        path_counts[cap_key] += 1

            # Link discovery — route discovered links to the right queue
            soup = BeautifulSoup(resp.text, "lxml")
            for link_tag in soup.find_all("a", href=True):
                href = link_tag["href"]
                absolute = urljoin(url, href)
                norm = normalize_url(absolute)

                if norm not in visited and not should_skip_url(norm):
                    visited.add(norm)
                    if is_priority_url(norm):
                        priority_queue.append(norm)
                    else:
                        normal_queue.append(norm)

            if fetch_count % 20 == 0:
                print(
                    f"Progress: {page_count}/{MAX_PAGES} saved | "
                    f"fetched={fetch_count} errors={errors} "
                    f"pq={len(priority_queue)} nq={len(normal_queue)} | "
                    f"skip: written={skip_already_written} short={skip_short_text} cap={skip_cap}",
                    file=sys.stderr,
                )

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
