"""
fetch_js_pages.py - Re-fetch JS-rendered pages that trafilatura couldn't extract.
==================================================================================
Uses Playwright (headless Chromium) to render pages then passes the live HTML to
trafilatura for clean text extraction.

Targets:
  - NASA Climate (12 skipped SPA pages)
  - Data.gov.in (10 skipped sector pages)

Run after: pip install playwright && playwright install chromium

Usage:
  python scripts/fetch_js_pages.py
  python scripts/fetch_js_pages.py --source nasa_climate
  python scripts/fetch_js_pages.py --source gov
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw" / "rag_sources"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pages that returned empty HTML with trafilatura's static fetch
# ---------------------------------------------------------------------------

NASA_SKIPPED = [
    {"url": "https://climate.nasa.gov/causes/",                "slug": "climate-change_causes"},
    {"url": "https://climate.nasa.gov/effects/",               "slug": "climate-change_effects"},
    {"url": "https://climate.nasa.gov/evidence/",              "slug": "climate-change_evidence"},
    {"url": "https://climate.nasa.gov/faq/",                   "slug": "climate-change_faq"},
    {"url": "https://climate.nasa.gov/",                       "slug": "climate-change_index"},
    {"url": "https://climate.nasa.gov/scientific-consensus/",  "slug": "climate-change_consensus"},
    {"url": "https://climate.nasa.gov/solutions/",             "slug": "climate-change_solutions"},
    {"url": "https://climate.nasa.gov/vital-signs/",           "slug": "vital-signs_index"},
    {"url": "https://climate.nasa.gov/vital-signs/arctic-sea-ice/",  "slug": "vital-signs_arctic-sea-ice"},
    {"url": "https://climate.nasa.gov/vital-signs/carbon-dioxide/",  "slug": "vital-signs_carbon-dioxide"},
    {"url": "https://climate.nasa.gov/vital-signs/global-temperature/", "slug": "vital-signs_global-temperature"},
    {"url": "https://climate.nasa.gov/vital-signs/ice-sheets/", "slug": "vital-signs_ice-sheets"},
]

GOV_SKIPPED = [
    {"url": "https://data.gov.in/",                          "slug": "index"},
    {"url": "https://data.gov.in/sector/agriculture",        "slug": "sector_agriculture"},
    {"url": "https://data.gov.in/sector/commerce-industry",  "slug": "sector_commerce-industry"},
    {"url": "https://data.gov.in/sector/education",          "slug": "sector_education"},
    {"url": "https://data.gov.in/sector/environment-forest-climate-change",
                                                             "slug": "sector_environment"},
    {"url": "https://data.gov.in/sector/finance",            "slug": "sector_finance"},
    {"url": "https://data.gov.in/sector/health-family-welfare",
                                                             "slug": "sector_health"},
    {"url": "https://data.gov.in/sector/labour-employment",  "slug": "sector_labour"},
    {"url": "https://data.gov.in/sector/science-technology", "slug": "sector_science"},
    {"url": "https://data.gov.in/sector/water-resources",    "slug": "sector_water"},
]


def fetch_rendered_html(url: str, page, timeout: int = 30000) -> str:
    """Navigate to URL with Playwright and return the full rendered HTML."""
    try:
        page.goto(url, wait_until="networkidle", timeout=timeout)
        # Extra wait for heavy JS SPAs
        page.wait_for_timeout(2000)
        return page.content()
    except Exception as e:
        log.warning("  Playwright timeout/error for %s: %s", url, e)
        # Try with domcontentloaded as fallback
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            page.wait_for_timeout(3000)
            return page.content()
        except Exception as e2:
            log.error("  Fallback also failed for %s: %s", url, e2)
            return ""


def save_html(html: str, out_dir: Path, slug: str, url: str) -> bool:
    """Save rendered HTML and meta.json. Returns True if extracted text is non-empty."""
    import trafilatura

    if not html or len(html.strip()) < 100:
        log.warning("  Empty HTML returned for %s", slug)
        return False

    # Save raw HTML
    html_path = out_dir / f"{slug}.html"
    html_path.write_text(html, encoding="utf-8")

    # Extract text
    extracted = trafilatura.extract(
        html,
        include_tables=True,
        favor_precision=True,
        no_fallback=False,
    )

    if not extracted or len(extracted.strip()) < 50:
        log.warning("  trafilatura still empty after Playwright render for %s", slug)
        html_path.unlink(missing_ok=True)  # don't keep empty files
        return False

    # Save meta.json
    from datetime import datetime, timezone
    meta = {
        "source_url": url,
        "fetch_date": datetime.now(timezone.utc).isoformat(),
        "page_title": slug.replace("_", " ").title(),
        "status_code": 200,
        "content_length": len(html),
        "fetch_method": "playwright+trafilatura",
    }
    meta_path = out_dir / f"{slug}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    char_count = len(extracted)
    log.info("  [OK] %s - extracted %d chars", slug, char_count)
    return True


def run_fetch(targets: list, out_dir: Path, source_name: str):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("playwright not installed. Run: pip install playwright && playwright install chromium")
        return

    try:
        import trafilatura  # noqa: F401
    except ImportError:
        log.error("trafilatura not installed")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("  Fetching %d JS-rendered pages for: %s", len(targets), source_name)
    log.info("=" * 60)

    success = 0
    failed  = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        for item in targets:
            url  = item["url"]
            slug = item["slug"]
            log.info("  Fetching: %s", url)
            html = fetch_rendered_html(url, page)
            ok   = save_html(html, out_dir, slug, url)
            if ok:
                success += 1
            else:
                failed.append(slug)
            time.sleep(1)  # polite delay between requests

        browser.close()

    log.info("")
    log.info("  Results for %s: %d/%d succeeded", source_name, success, len(targets))
    if failed:
        log.warning("  Failed slugs: %s", failed)
    log.info("  Output dir: %s", out_dir)
    return success, failed


def main():
    parser = argparse.ArgumentParser(
        description="Fetch JS-rendered pages with Playwright for NASA and Gov RAG sources"
    )
    parser.add_argument(
        "--source", default="all",
        choices=["nasa_climate", "gov", "all"],
        help="Which source to fetch (default: all)"
    )
    args = parser.parse_args()

    if args.source in ("all", "nasa_climate"):
        run_fetch(NASA_SKIPPED, RAW / "nasa_climate", "nasa_climate")

    if args.source in ("all", "gov"):
        run_fetch(GOV_SKIPPED, RAW / "gov", "gov")

    log.info("")
    log.info("Done. Now re-run Section 5 to re-chunk the newly fetched pages:")
    log.info("  python scripts/clean_datasets.py --section 5")


if __name__ == "__main__":
    main()
