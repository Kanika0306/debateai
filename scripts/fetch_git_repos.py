"""
fetch_git_repos.py - Clone GitHub/GitLab repos for debate-ai.
"""
import os
import sys
import subprocess
import json
import requests
from datetime import datetime, timezone

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE, "data", "raw")

results = []


def clone_repo(url, dest, name):
    """Clone a git repo into dest."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.listdir(dest):
        print(f"  [SKIP] {name} -- directory already exists and is non-empty")
        return True

    # Setup environment to prevent hangs (credential prompts) and skip Git LFS smudge
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    try:
        # If dest exists but is empty, clone into it
        if os.path.exists(dest):
            # Clone into a temp name then move
            subprocess.run(
                ["git", "clone", "--depth", "1", url, dest + "_tmp"],
                check=True, capture_output=True, text=True, env=env
            )
            # Move contents
            import shutil
            for item in os.listdir(dest + "_tmp"):
                shutil.move(os.path.join(dest + "_tmp", item), os.path.join(dest, item))
            shutil.rmtree(dest + "_tmp")
        else:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, dest],
                check=True, capture_output=True, text=True, env=env
            )
        print(f"  [OK] Cloned {name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL] Failed to clone {name}: {e.stderr}")
        return False


def fetch_fakenewsnet():
    """Clone FakeNewsNet and inspect its structure."""
    print(f"\n{'='*60}")
    print("  FakeNewsNet")
    print(f"{'='*60}")

    dest = os.path.join(DATA_RAW, "misinformation", "fakenewsnet")
    url = "https://github.com/KaiDMML/FakeNewsNet"

    success = clone_repo(url, dest, "FakeNewsNet")

    if success:
        # Inspect structure
        print("\n  Repository structure:")
        for root, dirs, files in os.walk(dest):
            # Skip .git
            dirs[:] = [d for d in dirs if d != ".git"]
            level = root.replace(dest, "").count(os.sep)
            indent = "    " * (level + 1)
            print(f"{indent}{os.path.basename(root)}/")
            if level < 2:  # Only show top 2 levels
                subindent = "    " * (level + 2)
                for f in files[:10]:
                    print(f"{subindent}{f}")
                if len(files) > 10:
                    print(f"{subindent}... and {len(files) - 10} more files")

        # Count any data files
        data_files = 0
        for root, dirs, files in os.walk(dest):
            dirs[:] = [d for d in dirs if d != ".git"]
            data_files += len([f for f in files if f.endswith(('.json', '.csv', '.tsv'))])

        readme_path = os.path.join(dest, "README.txt")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"FakeNewsNet\n")
            f.write(f"Cloned: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"Source: {url}\n\n")
            f.write("NOTE: This repo contains fetch scripts + IDs, not full article content.\n")
            f.write("To fetch actual articles, you need to:\n")
            f.write("  1. Install requirements: pip install -r requirements.txt\n")
            f.write("  2. Set up Twitter API credentials (for tweet data)\n")
            f.write("  3. Run the data collection script per the repo's README\n")
            f.write(f"\nData files found in repo: {data_files}\n")

        results.append({
            "name": "FakeNewsNet",
            "status": "success (IDs only, articles need manual fetch)",
            "files": data_files,
            "path": dest,
        })
        print(f"\n  Note: Repo contains fetch scripts + IDs. Full articles require")
        print(f"  running their collection scripts with Twitter API credentials.")
        print(f"  Data files found in repo: {data_files}")
    else:
        results.append({
            "name": "FakeNewsNet",
            "status": "failed",
            "path": dest,
        })


def fetch_checkthat():
    """Find and clone the most recent CheckThat Lab claim-detection repo."""
    print(f"\n{'='*60}")
    print("  CheckThat Lab (Claim Detection)")
    print(f"{'='*60}")

    dest = os.path.join(DATA_RAW, "claim_detection", "claimbuster")

    # Try GitLab API to find projects
    try:
        # Get group info first
        resp = requests.get(
            "https://gitlab.com/api/v4/groups/checkthat_lab/projects",
            params={"per_page": 100, "order_by": "created_at", "sort": "desc"},
            timeout=30,
        )
        resp.raise_for_status()
        projects = resp.json()

        # Find claim-detection related repos
        claim_projects = []
        for p in projects:
            name_lower = p["name"].lower()
            desc_lower = (p.get("description") or "").lower()
            if any(kw in name_lower or kw in desc_lower for kw in
                   ["claim", "check-worthiness", "task1", "task-1"]):
                claim_projects.append(p)

        if not claim_projects:
            # Fallback: just take all projects and pick most recent
            claim_projects = projects

        print(f"  Found {len(claim_projects)} relevant projects:")
        for p in claim_projects[:5]:
            print(f"    - {p['name']} (created: {p.get('created_at', 'unknown')[:10]})")
            print(f"      {p.get('web_url', '')}")

        # Pick the most recent one
        chosen = claim_projects[0]
        clone_url = chosen.get("http_url_to_repo", chosen.get("web_url") + ".git")
        print(f"\n  Selected: {chosen['name']}")
        print(f"  URL: {clone_url}")

        success = clone_repo(clone_url, dest, chosen["name"])

        if success:
            readme_path = os.path.join(dest, "README_download.txt")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(f"CheckThat Lab - Claim Detection Data\n")
                f.write(f"Cloned: {datetime.now(timezone.utc).isoformat()}\n")
                f.write(f"Project: {chosen['name']}\n")
                f.write(f"Source: {clone_url}\n")
                f.write(f"Year: {chosen.get('created_at', 'unknown')[:4]}\n")

            results.append({
                "name": f"CheckThat Lab ({chosen['name']})",
                "status": "success",
                "year": chosen.get("created_at", "unknown")[:4],
                "path": dest,
            })
        else:
            results.append({
                "name": "CheckThat Lab",
                "status": "failed (clone error)",
                "path": dest,
            })

    except Exception as e:
        print(f"  [FAIL] GitLab API failed: {e}")
        # Fallback: try known repos
        fallback_urls = [
            "https://gitlab.com/checkthat_lab/clef2024-checkthat-lab.git",
            "https://gitlab.com/checkthat_lab/clef2023-checkthat-lab.git",
        ]
        cloned = False
        for url in fallback_urls:
            print(f"  Trying fallback: {url}")
            if clone_repo(url, dest, "CheckThat Lab"):
                year = "2024" if "2024" in url else "2023"
                results.append({
                    "name": f"CheckThat Lab ({year})",
                    "status": "success (fallback)",
                    "year": year,
                    "path": dest,
                })
                cloned = True
                break
        if not cloned:
            results.append({
                "name": "CheckThat Lab",
                "status": "failed",
                "error": str(e),
                "path": dest,
            })


def main():
    fetch_fakenewsnet()
    fetch_checkthat()

    print(f"\n\n{'='*60}")
    print("  GIT REPOS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['name']}: {r['status']} -> {r['path']}")

    results_path = os.path.join(DATA_RAW, "git_repos_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
