"""
Respectful scraper for Bloom's questions. Defaults to conservative settings and may return zero rows.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
from urllib import robotparser

import requests
import trafilatura
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from src.utils.io import ensure_dir, utcnow, write_jsonl


@dataclass
class ScrapeItem:
    id: str
    text: str
    labels: Optional[List[int]]
    source_url: str
    source_title: str
    license: str
    timestamp: str
    split_hint: str


SAFE_SITES = ["https://oercommons.org", "https://open.umn.edu", "https://github.com"]
ALLOWED_LICENSES = {"cc-by", "cc-by-sa", "cc0", "public-domain"}


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def can_fetch(url: str, user_agent: str = "blooms-bot") -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return False


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return ""


def scrape_page(url: str, timeout: float) -> Optional[ScrapeItem]:
    if not can_fetch(url):
        return None
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "blooms-bot"})
        resp.raise_for_status()
    except Exception:
        return None
    html = resp.text
    cleaned = trafilatura.extract(html)
    if not cleaned:
        return None
    title = extract_title(html)
    item = ScrapeItem(
        id=hash_text(cleaned),
        text=cleaned.strip(),
        labels=None,
        source_url=url,
        source_title=title,
        license="unknown",
        timestamp=utcnow(),
        split_hint="train",
    )
    return item


def search_sites(query: str, max_pages: int, sites: List[str]) -> List[str]:
    urls: List[str] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_pages):
            url = r.get("href") or r.get("link")
            if not url:
                continue
            if sites and not any(url.startswith(site) for site in sites):
                continue
            urls.append(url)
    return urls


def main():
    parser = argparse.ArgumentParser(description="Respectful scraper")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--sites", nargs="*", default=SAFE_SITES)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--rate-limit", type=float, default=1.0)
    parser.add_argument("--allow-licenses", type=str, default=",".join(ALLOWED_LICENSES))
    args = parser.parse_args()

    allow_licenses = {x.strip() for x in args.allow_licenses.split(",") if x}
    ensure_dir("data/raw")
    ensure_dir("outputs/metrics")

    urls = search_sites("Bloom taxonomy questions", args.max_pages, args.sites)
    records: List[ScrapeItem] = []
    raw_dir = Path("data/raw")
    for url in urls:
        time.sleep(max(args.rate_limit, 0.1))
        item = scrape_page(url, args.timeout)
        if not item:
            continue
        if item.license and allow_licenses and item.license.lower() not in allow_licenses and item.license != "unknown":
            continue
        records.append(item)
        html_path = raw_dir / f"{item.id}.html"
        html_path.write_text(item.text, encoding="utf-8")

    out_jsonl = raw_dir / "scrape.jsonl"
    write_jsonl(out_jsonl, [item.__dict__ for item in records])

    prov_path = Path("outputs/metrics/provenance.csv")
    ensure_dir(prov_path.parent)
    prov_lines = ["id,source_url,license,timestamp"] + [f"{r.id},{r.source_url},{r.license},{r.timestamp}" for r in records]
    prov_path.write_text("\n".join(prov_lines), encoding="utf-8")

    card = Path("outputs/metrics/dataset_card.md")
    card.write_text(
        "\n".join(
            [
                "# Dataset card (scrape)",
                f"Total scraped: {len(records)}",
                f"Allowed licenses: {sorted(allow_licenses)}",
                "Scraping is conservative; synthetic fallback will be used if insufficient data.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Scraped {len(records)} pages -> {out_jsonl}")


if __name__ == "__main__":
    main()
