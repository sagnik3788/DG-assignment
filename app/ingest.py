"""Bounded, polite refresh of public Wasalt sitemaps and DarGlobal public pages."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
import re
import sys
import time
from itertools import zip_longest
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .listings import save_listing_records

DATA_PATH = Path(os.getenv("DATA_PATH", "data/documents.jsonl"))
LIMIT = max(1, int(os.getenv("MAX_PAGES_PER_RUN", "160")))
UA = "GulfPropertyIntelligence/1.0 (public-page research demo)"
ROBOTS = {"wasalt.sa": "https://wasalt.sa/robots.txt", "darglobal.co.uk": "https://darglobal.co.uk/robots.txt"}
DAR_SITEMAP = "https://darglobal.co.uk/sitemap.xml"
DAR_SEEDS = [
    "https://darglobal.co.uk/", "https://darglobal.co.uk/about",
    "https://darglobal.co.uk/projects", "https://darglobal.co.uk/aida",
    "https://darglobal.co.uk/projects/properties-in-qatar",
    "https://darglobal.co.uk/projects/properties-in-spain",
    "https://darglobal.co.uk/projects/properties-in-uae",
    "https://darglobal.co.uk/projects/properties-in-saudi-arabia/villas",
]
SITEMAP_SOURCES = [
    "https://cdn.wasalt.sa/sitemap/static_sitemap_en_sa.xml.gz",
    "https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz",
    "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz",
    "https://cdn.wasalt.sa/sitemap/rental_pdp_sitemap_en_sa.xml.gz",
    "https://cdn.wasalt.sa/sitemap/static_sitemap_ar_sa.xml.gz",
    "https://cdn.wasalt.sa/sitemap/category_sitemap_ar_sa.xml.gz",
    "https://cdn.wasalt.sa/sitemap/product_sitemap_ar_sa.xml.gz",
    "https://cdn.wasalt.sa/sitemap/rental_pdp_sitemap_ar_sa.xml.gz",
]


def fetch_sitemap(url: str) -> list[str]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=25)
        r.raise_for_status()
        body = gzip.decompress(r.content) if r.content[:2] == b"\x1f\x8b" else r.content
        root = ET.fromstring(body)
        return [node.text.strip() for node in root.findall(".//{*}loc") if node.text]
    except Exception as exc:
        print(f"Skipping sitemap {url}: {exc}", file=sys.stderr)
        return []


def candidate_urls() -> tuple[list[str], list[str]]:
    maps = {url: fetch_sitemap(url) for url in SITEMAP_SOURCES}
    dar = list(dict.fromkeys(DAR_SEEDS + fetch_sitemap(DAR_SITEMAP)))
    dar.sort(key=lambda u: (0 if "/projects/" in u else 1 if "/press/" in u else 2 if "/blog/" in u else 3, u))
    static = []
    for u in maps[SITEMAP_SOURCES[0]] + maps[SITEMAP_SOURCES[4]]:
        if not any(term in u.lower() for term in ("privacy", "terms", "policy")):
            static.append(u)
    # Rotate through the entire published category and property sitemaps across runs.
    groups = [maps[url] for url in (SITEMAP_SOURCES[1], SITEMAP_SOURCES[5],
             SITEMAP_SOURCES[2], SITEMAP_SOURCES[6], SITEMAP_SOURCES[3], SITEMAP_SOURCES[7])]
    # Round-robin categories, sale, and rental URLs in both languages so early runs cover each family.
    rotating = [u for row in zip_longest(*groups) for u in row if u]
    # Prefer company/project/press/blog information over arbitrary sitemap order.
    dar = [u for u in dar if not any(x in u.lower() for x in ("/pay-online", "/success", "/thank-you", "/privacy", "/terms", "/careers", "/internship"))]
    # The run loop rotates this DarGlobal queue while preserving Wasalt's static URLs.
    core = list(dict.fromkeys(dar + static))
    return core, list(dict.fromkeys(rotating))


def allowed(url: str, rules: dict[str, list[str]]) -> bool:
    host = urlparse(url).hostname or ""
    for domain, disallowed in rules.items():
        if host == domain or host.endswith("." + domain):
            path = urlparse(url).path
            return not any(path.startswith(prefix) for prefix in disallowed)
    return False


def robots_rules() -> dict[str, list[str]]:
    rules = {}
    for domain, url in ROBOTS.items():
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
            r.raise_for_status()
            # Use the default user-agent group only. Sitemap paths are public; disallows are honored.
            lines = r.text.splitlines()
            if not any(line.strip().lower().startswith("user-agent:") for line in lines):
                raise ValueError("robots endpoint did not return usable directives")
            active, disallowed = False, []
            for line in lines:
                line = line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, value = [part.strip() for part in line.split(":", 1)]
                if key.lower() == "user-agent":
                    if active and value != "*":
                        break
                    active = value == "*"
                elif active and key.lower() == "disallow" and value:
                    disallowed.append(value)
            rules[domain] = disallowed
        except Exception as exc:
            print(f"Cannot read robots policy for {domain}; skipping its pages: {exc}", file=sys.stderr)
            rules[domain] = ["/"]
    return rules


def extract(url: str, session: requests.Session) -> dict | bool | None:
    try:
        r = session.get(url, headers={"User-Agent": UA}, timeout=20)
        if r.status_code in (401, 403, 429):
            print(f"Access limited ({r.status_code}); skipped {url}", file=sys.stderr)
            return False
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        if any(s in title.lower() for s in ("just a moment", "access denied", "attention required")):
            print(f"Bot-check page; skipped {url}", file=sys.stderr)
            return False
        for node in soup.select("script,style,noscript,svg,nav,footer,header,form,iframe"):
            node.decompose()
        main = soup.select_one("main") or soup.select_one("article") or soup.body or soup
        text = re.sub(r"\s+", " ", main.get_text(" ", strip=True)).strip()
        if len(text) < 100:
            return None
        text = text[:24000]
        language = "ar" if "/en/" not in url and re.search(r"[\u0600-\u06ff]", text) else "en"
        return {"url": url, "site": "Wasalt" if "wasalt.sa" in url else "DarGlobal",
                "title": title[:300] or url, "language": language, "text": text,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                "last_modified": r.headers.get("Last-Modified")}
    except Exception as exc:
        print(f"Fetch failed for {url}: {exc}", file=sys.stderr)
        return None


def run() -> int:
    rules = robots_rules()
    core, rotating = candidate_urls()
    state_path = DATA_PATH.parent / "crawl_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    old = {}
    if DATA_PATH.exists():
        for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
            try:
                doc = json.loads(line)
                old[doc["url"]] = doc
            except (ValueError, KeyError):
                continue
    last = state.get("last_url")
    offset = rotating.index(last) + 1 if last in rotating else 0
    dar_urls = [u for u in core if urlparse(u).hostname == "darglobal.co.uk"]
    static = [u for u in core if urlparse(u).hostname != "darglobal.co.uk"]
    last_dar = state.get("last_dar_url") or next((u["url"] for u in reversed(list(old.values())) if u.get("site") == "DarGlobal"), None)
    dar_offset = dar_urls.index(last_dar) + 1 if last_dar in dar_urls else 0
    dar_take = min(32, LIMIT)
    selected_dar = dar_urls[dar_offset:dar_offset + dar_take]
    if len(selected_dar) < dar_take and dar_urls:
        selected_dar += dar_urls[:dar_take - len(selected_dar)]
    core = list(dict.fromkeys(selected_dar + static))
    take = max(0, LIMIT - len(core))
    selected = rotating[offset:offset + take]
    if len(selected) < take and rotating:
        selected += rotating[:take - len(selected)]
    retry = [u for u in state.get("retry_urls", []) if u not in selected]
    urls = list(dict.fromkeys(retry[:take] + core + selected))[:LIMIT]
    session = requests.Session()
    updated = dict(old)
    blocked_hosts, failed = set(), []
    fetched = 0
    last_successful = last
    last_dar_successful = last_dar
    for url in urls:
        host = urlparse(url).hostname or ""
        if host in blocked_hosts or not allowed(url, rules):
            continue
        item = extract(url, session)
        if item:
            previous = old.get(url)
            if previous and previous.get("content_hash") == item["content_hash"]:
                item["fetched_at"] = previous.get("fetched_at", item["fetched_at"])
            updated[url] = item
            fetched += 1
            last_successful = url if url in rotating else last_successful
            last_dar_successful = url if host == "darglobal.co.uk" else last_dar_successful
        else:
            failed.append(url)
            # Stop this host only on an explicit access block/challenge; ordinary parse/network failures are per-page.
            if item is False:
                blocked_hosts.add(host)
        time.sleep(0.35)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA_PATH.with_suffix(DATA_PATH.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in updated.values()), encoding="utf-8")
    tmp.replace(DATA_PATH)
    listing_count = save_listing_records(list(updated.values()), DATA_PATH.with_name("listings.jsonl"))
    state_path.write_text(json.dumps({"last_url": last_successful, "last_dar_url": last_dar_successful,
                                      "retry_urls": failed[:500]}, indent=2), encoding="utf-8")
    print(f"Fetched {fetched} usable pages; {len(failed)} deferred; corpus has {len(updated)} documents and {listing_count} structured listings at {DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
