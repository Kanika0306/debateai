"""
fetch_rag_sources.py - Fetch curated web pages for RAG knowledge base.
Topics: economy, health, climate.
Saves raw HTML + meta JSON for each page.
"""
import os
import sys
import json

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import re
import time
import requests
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
from html.parser import HTMLParser

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE, "data", "raw", "rag_sources")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


class TitleParser(HTMLParser):
    """Extract <title> from HTML."""
    def __init__(self):
        super().__init__()
        self._in_title = False
        self.title = ""

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "title":
            self._in_title = True

    def handle_data(self, data):
        if self._in_title:
            self.title += data

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self._in_title = False


def slugify(url):
    """Create a filesystem-safe slug from a URL."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    query = parsed.query.replace("&", "_").replace("=", "-")
    slug = f"{path}_{query}" if query else path
    slug = re.sub(r'[^\w\-]', '_', slug)
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug[:100] or "index"


def get_title(html):
    """Extract page title from HTML."""
    parser = TitleParser()
    try:
        parser.feed(html)
    except:
        pass
    return parser.title.strip() or "Untitled"


def fetch_page(url, dest_dir, delay=1.5):
    """Fetch a single page and save HTML + meta JSON."""
    slug = slugify(url)
    html_path = os.path.join(dest_dir, f"{slug}.html")
    meta_path = os.path.join(dest_dir, f"{slug}.meta.json")

    # Skip if already fetched
    if os.path.exists(html_path):
        print(f"    [SKIP] {slug} (already exists)")
        return True

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        html = resp.text

        # Save HTML
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        # Save meta
        meta = {
            "source_url": url,
            "fetch_date": datetime.now(timezone.utc).isoformat(),
            "page_title": get_title(html),
            "status_code": resp.status_code,
            "content_length": len(html),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"    [OK] {slug} ({len(html)} bytes) -- {meta['page_title'][:60]}")
        time.sleep(delay)  # Be polite
        return True

    except Exception as e:
        print(f"    [FAIL] {slug} -- {e}")
        return False


# ──────────────────────────────────────────────
# Curated URL lists per domain
# ──────────────────────────────────────────────

GOV_URLS = [
    "https://www.data.gov.in/",
    "https://www.data.gov.in/sector/health-family-welfare",
    "https://www.data.gov.in/sector/finance",
    "https://www.data.gov.in/sector/commerce-industry",
    "https://www.data.gov.in/sector/environment-forest-climate-change",
    "https://www.data.gov.in/sector/agriculture",
    "https://www.data.gov.in/sector/labour-employment",
    "https://www.data.gov.in/catalogs",
    "https://www.data.gov.in/sector/education",
    "https://www.data.gov.in/sector/science-technology",
    "https://www.data.gov.in/sector/water-resources",
]

WHO_URLS = [
    "https://www.who.int/",
    "https://www.who.int/news-room/fact-sheets/detail/climate-change-and-health",
    "https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases",
    "https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response",
    "https://www.who.int/news-room/fact-sheets/detail/diabetes",
    "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    "https://www.who.int/news-room/fact-sheets/detail/air-pollution",
    "https://www.who.int/health-topics/air-pollution",
    "https://www.who.int/health-topics/climate-change",
    "https://www.who.int/health-topics/universal-health-coverage",
    "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight",
    "https://www.who.int/news-room/fact-sheets/detail/tobacco",
    "https://www.who.int/news-room/fact-sheets/detail/drinking-water",
    "https://www.who.int/data/gho",
    "https://www.who.int/publications/i",
]

WORLDBANK_URLS = [
    "https://data.worldbank.org/",
    "https://data.worldbank.org/topic/economy-and-growth",
    "https://data.worldbank.org/topic/health",
    "https://data.worldbank.org/topic/climate-change",
    "https://data.worldbank.org/topic/poverty",
    "https://data.worldbank.org/topic/education",
    "https://data.worldbank.org/indicator/NY.GDP.MKTP.CD",
    "https://data.worldbank.org/indicator/SP.DYN.LE00.IN",
    "https://data.worldbank.org/indicator/EN.ATM.CO2E.KT",
    "https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS",
    "https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG",
    "https://data.worldbank.org/indicator/SH.XPD.CHEX.GD.ZS",
    "https://data.worldbank.org/country",
]

NASA_URLS = [
    "https://climate.nasa.gov/",
    "https://climate.nasa.gov/evidence/",
    "https://climate.nasa.gov/causes/",
    "https://climate.nasa.gov/effects/",
    "https://climate.nasa.gov/scientific-consensus/",
    "https://climate.nasa.gov/solutions/",
    "https://climate.nasa.gov/faq/",
    "https://climate.nasa.gov/vital-signs/",
    "https://climate.nasa.gov/vital-signs/carbon-dioxide/",
    "https://climate.nasa.gov/vital-signs/global-temperature/",
    "https://climate.nasa.gov/vital-signs/ice-sheets/",
    "https://sealevel.nasa.gov",
    "https://climate.nasa.gov/vital-signs/arctic-sea-ice/",
]


def fetch_domain(name, urls, dest_subdir):
    """Fetch all URLs for a domain."""
    dest = os.path.join(DATA_RAW, dest_subdir)
    os.makedirs(dest, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {name} — fetching {len(urls)} pages")
    print(f"{'='*60}")

    success = 0
    failed = 0
    for url in urls:
        if fetch_page(url, dest):
            success += 1
        else:
            failed += 1

    print(f"\n  {name}: {success} fetched, {failed} failed")
    return {"name": name, "fetched": success, "failed": failed, "path": dest}


def main():
    results = []

    results.append(fetch_domain("Data.gov.in", GOV_URLS, "gov"))
    results.append(fetch_domain("WHO", WHO_URLS, "who"))
    results.append(fetch_domain("World Bank", WORLDBANK_URLS, "worldbank"))
    results.append(fetch_domain("NASA Climate", NASA_URLS, "nasa_climate"))

    print(f"\n\n{'='*60}")
    print("  RAG SOURCES SUMMARY")
    print(f"{'='*60}")
    print(f"{'Domain':<20} {'Fetched':<10} {'Failed':<10} {'Path'}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<20} {r['fetched']:<10} {r['failed']:<10} {r['path']}")

    results_path = os.path.join(DATA_RAW, "rag_sources_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
