"""
verify_apis.py - Test one query to each API endpoint for debate-ai.
Does NOT download bulk data - just confirms reachability + response shape.
"""
import os
import sys
import json
import requests
from datetime import datetime, timezone

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Try loading .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE, ".env"))
except:
    pass

results = []


def test_semantic_scholar():
    """Test Semantic Scholar API - no key required for basic tier."""
    print(f"\n{'='*60}")
    print("  Semantic Scholar API")
    print(f"{'='*60}")

    import time
    for attempt in range(3):
        try:
            resp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": "inflation economics", "limit": 3},
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  Rate limited (429), waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            print(f"  Status: {resp.status_code}")
            print(f"  Response keys: {list(data.keys())}")
            print(f"  Total results: {data.get('total', 'N/A')}")
            if "data" in data and data["data"]:
                print(f"  First result keys: {list(data['data'][0].keys())}")
                print(f"  First title: {data['data'][0].get('title', 'N/A')[:80]}")

            results.append({
                "api": "Semantic Scholar",
                "status": "success",
                "response_shape": {k: type(v).__name__ for k, v in data.items()},
                "total": data.get("total"),
            })
            return
        except Exception as e:
            if attempt == 2:
                print(f"  [FAIL] Failed after retries: {e}")
                results.append({"api": "Semantic Scholar", "status": f"failed: {e}"})
            else:
                time.sleep(3)


def test_pubmed():
    """Test PubMed E-utilities - no key required for low volume."""
    print(f"\n{'='*60}")
    print("  PubMed E-utilities")
    print(f"{'='*60}")

    try:
        # ESearch
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": "inflation health effects",
                "retmax": 3,
                "retmode": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        esearch = data.get("esearchresult", {})
        print(f"  Status: {resp.status_code}")
        print(f"  Response keys: {list(data.keys())}")
        print(f"  Search result keys: {list(esearch.keys())}")
        print(f"  Count: {esearch.get('count', 'N/A')}")
        print(f"  IDs returned: {esearch.get('idlist', [])[:5]}")

        results.append({
            "api": "PubMed E-utilities",
            "status": "success",
            "count": esearch.get("count"),
            "sample_ids": esearch.get("idlist", [])[:3],
        })
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        results.append({"api": "PubMed E-utilities", "status": f"failed: {e}"})


def test_wikipedia():
    """Test Wikipedia API - no key required."""
    print(f"\n{'='*60}")
    print("  Wikipedia API")
    print(f"{'='*60}")

    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": "inflation economics",
                "srlimit": 3,
                "format": "json",
            },
            headers={
                "User-Agent": "debate-ai-research-bot/1.0 (https://github.com/debateai; contact@debateai.dev)",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        query = data.get("query", {})
        search = query.get("search", [])
        print(f"  Status: {resp.status_code}")
        print(f"  Response keys: {list(data.keys())}")
        print(f"  Search results: {len(search)}")
        if search:
            print(f"  First result keys: {list(search[0].keys())}")
            print(f"  First title: {search[0].get('title', 'N/A')}")

        results.append({
            "api": "Wikipedia",
            "status": "success",
            "results_count": len(search),
            "first_title": search[0].get("title") if search else None,
        })
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        results.append({"api": "Wikipedia", "status": f"failed: {e}"})


def test_google_factcheck():
    """Test Google Fact Check Tools API - requires API key."""
    print(f"\n{'='*60}")
    print("  Google Fact Check Tools API")
    print(f"{'='*60}")

    api_key = os.environ.get("GOOGLE_FACTCHECK_API_KEY")
    if not api_key:
        print("  [SKIP] GOOGLE_FACTCHECK_API_KEY not set in .env")
        results.append({
            "api": "Google Fact Check",
            "status": "skipped (no API key)",
        })
        return

    try:
        resp = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": "inflation", "key": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        claims = data.get("claims", [])
        print(f"  Status: {resp.status_code}")
        print(f"  Response keys: {list(data.keys())}")
        print(f"  Claims found: {len(claims)}")
        if claims:
            print(f"  First claim keys: {list(claims[0].keys())}")
            print(f"  First claim text: {claims[0].get('text', 'N/A')[:80]}")

        results.append({
            "api": "Google Fact Check",
            "status": "success",
            "claims_count": len(claims),
        })
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        results.append({"api": "Google Fact Check", "status": f"failed: {e}"})


def test_newsapi():
    """Test NewsAPI - requires API key."""
    print(f"\n{'='*60}")
    print("  NewsAPI")
    print(f"{'='*60}")

    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        print("  [SKIP] NEWSAPI_KEY not set in .env")
        results.append({"api": "NewsAPI", "status": "skipped (no API key)"})
        return

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": "inflation",
                "pageSize": 3,
                "apiKey": api_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        articles = data.get("articles", [])
        print(f"  Status: {resp.status_code}")
        print(f"  Response keys: {list(data.keys())}")
        print(f"  Total results: {data.get('totalResults', 'N/A')}")
        print(f"  Articles returned: {len(articles)}")
        if articles:
            print(f"  First article keys: {list(articles[0].keys())}")
            print(f"  First title: {articles[0].get('title', 'N/A')[:80]}")

        results.append({
            "api": "NewsAPI",
            "status": "success",
            "total_results": data.get("totalResults"),
        })
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        results.append({"api": "NewsAPI", "status": f"failed: {e}"})


def main():
    # Test keyless APIs first
    test_semantic_scholar()
    test_pubmed()
    test_wikipedia()

    # Test keyed APIs
    test_google_factcheck()
    test_newsapi()

    # Summary
    print(f"\n\n{'='*60}")
    print("  API VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'API':<30} {'Status'}")
    print("-" * 60)
    for r in results:
        print(f"{r['api']:<30} {r['status']}")

    results_path = os.path.join(BASE, "data", "raw", "api_verification_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
