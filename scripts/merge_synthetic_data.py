"""
merge_synthetic_data.py
==========================
Merges reviewed synthetic_out/ files into data/processed/, with:
  1. Near-duplicate dedup (text hashing, not just exact-match)
  2. RAG chunks -> separate `rag_synthetic` namespace/index, NOT merged
     into the same index as real WHO/NASA/gov chunks
  3. Fallacy examples -> merged directly (labels are self-contained)
  4. Eval set -> exports a flagged review CSV for manual verification.

USAGE:
  python merge_synthetic_data.py rag --domain gov
  python merge_synthetic_data.py rag --domain worldbank
  python merge_synthetic_data.py fallacies
  python merge_synthetic_data.py eval_review     # writes CSV, does not merge
  python merge_synthetic_data.py eval_commit      # after review, merges approved rows only
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

SYNTH_DIR = Path("synthetic_out")
DATA_DIR = Path("data/processed")

RAG_REAL_DIR = DATA_DIR / "rag_chunks"
RAG_SYNTH_DIR = DATA_DIR / "rag_synthetic"          # separate namespace
FALLACY_FILE = DATA_DIR / "fallacies" / "fallacy_examples_unified.jsonl"
EVAL_FILE = DATA_DIR / "eval_set.jsonl"
EVAL_REVIEW_CSV = SYNTH_DIR / "eval_set_review.csv"
EVAL_APPROVED_CSV = SYNTH_DIR / "eval_set_review.csv"  # same file, with "approve" column filled


def normalize_text(text):
    """Lowercase, strip punctuation/whitespace variance, for dedup hashing."""
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t


def text_hash(text):
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def load_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records, dry_run=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"  [dry-run] would write {len(records)} records -> {path}")
        return
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records)} records -> {path}")


# ---------------------------------------------------------------------------
# 1. RAG CHUNKS -> separate namespace, dedup against existing synthetic file
# ---------------------------------------------------------------------------

def merge_rag(domain, dry_run=False):
    synth_path = SYNTH_DIR / f"{domain}_chunks_synthetic.jsonl"
    real_path = RAG_REAL_DIR / f"{domain}_chunks.jsonl"
    out_path = RAG_SYNTH_DIR / f"{domain}_chunks_synthetic.jsonl"

    synth_records = load_jsonl(synth_path)
    real_records = load_jsonl(real_path)
    existing_synth = load_jsonl(out_path)

    if not synth_records:
        print(f"No synthetic records found at {synth_path}, skipping.")
        return

    real_hashes = {text_hash(r.get("text", "")) for r in real_records}
    existing_hashes = {text_hash(r.get("text", "")) for r in existing_synth}

    kept, dropped_dupe, warned_near_real = [], 0, 0
    seen_this_batch = set()

    for r in synth_records:
        h = text_hash(r.get("text", ""))
        if h in seen_this_batch or h in existing_hashes:
            dropped_dupe += 1
            continue
        if h in real_hashes:
            warned_near_real += 1
            print(f"  [warn] near-duplicate of REAL chunk, dropping: {r.get('chunk_id')}")
            continue
        seen_this_batch.add(h)
        kept.append(r)

    print(f"\n[{domain}] {len(synth_records)} synthetic -> {len(kept)} kept "
          f"({dropped_dupe} internal dupes, {warned_near_real} matched real corpus)")

    merged = existing_synth + kept
    write_jsonl(out_path, merged, dry_run)
    print(f"  NOTE: written to rag_synthetic/, NOT rag_chunks/ — rebuild a "
          f"separate index for this if you want it queryable, and keep it "
          f"OUT of the NLI training evidence pool.")


# ---------------------------------------------------------------------------
# 2. FALLACY EXAMPLES -> merge directly, dedup by normalized text
# ---------------------------------------------------------------------------

def merge_fallacies(dry_run=False):
    synth_path = SYNTH_DIR / "fallacy_examples_synthetic.jsonl"
    synth_records = load_jsonl(synth_path)
    existing = load_jsonl(FALLACY_FILE)

    if not synth_records:
        print(f"No synthetic records found at {synth_path}, skipping.")
        return

    existing_hashes = {text_hash(r.get("text", "")) for r in existing}
    kept, dropped = [], 0
    seen = set()

    for r in synth_records:
        h = text_hash(r.get("text", ""))
        if h in seen or h in existing_hashes:
            dropped += 1
            continue
        seen.add(h)
        kept.append(r)

    print(f"\n[fallacies] {len(synth_records)} synthetic -> {len(kept)} kept "
          f"({dropped} dupes dropped)")

    merged = existing + kept
    write_jsonl(FALLACY_FILE, merged, dry_run)

    from collections import Counter
    counts = Counter(r.get("label") or r.get("fallacy_type") for r in merged)
    print("\n  Updated class distribution:")
    for label, c in counts.most_common():
        print(f"    {label}: {c}")


# ---------------------------------------------------------------------------
# 3. EVAL SET -> export to CSV for manual review, never auto-merge
# ---------------------------------------------------------------------------

def eval_review(dry_run=False):
    synth_path = SYNTH_DIR / "eval_set_synthetic.jsonl"
    records = load_jsonl(synth_path)
    if not records:
        print(f"No synthetic records found at {synth_path}, skipping.")
        return

    rows = []
    for r in records:
        claims = r.get("gold_claims", [])
        claim_summary = " | ".join(
            f"{c.get('claim_text','')} -> {c.get('verdict','')}" for c in claims
        )
        fallacy = r.get("gold_fallacy") or {}
        rows.append({
            "id": r.get("id"),
            "topic": r.get("topic"),
            "transcript": r.get("transcript", "").replace("\n", " / "),
            "claim_and_verdict": claim_summary,
            "fallacy_type": fallacy.get("type", ""),
            "fallacy_span": fallacy.get("span", ""),
            "verdict_plausible_given_real_index": "",  # fill: yes/no/unsure
            "approve": "",  # fill: yes/no
            "notes": "",
        })

    if dry_run:
        print(f"[dry-run] would write {len(rows)} rows -> {EVAL_REVIEW_CSV}")
        return

    EVAL_REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_REVIEW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {EVAL_REVIEW_CSV}")
    print("Open this in Excel/Sheets. For each row:")
    print("  - fill 'verdict_plausible_given_real_index' (yes/no/unsure)")
    print("  - fill 'approve' with yes/no")
    print("Then run: python merge_synthetic_data.py eval_commit")


def eval_commit(dry_run=False):
    if not EVAL_APPROVED_CSV.exists():
        print(f"No review CSV found at {EVAL_APPROVED_CSV}. Run eval_review first.")
        return

    synth_records = {r["id"]: r for r in load_jsonl(SYNTH_DIR / "eval_set_synthetic.jsonl")}
    approved_ids = []
    total_rows = 0

    with open(EVAL_APPROVED_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if row.get("approve", "").strip().lower() == "yes":
                approved_ids.append(row["id"])

    if total_rows == 0:
        print("Review CSV is empty.")
        return

    approved_records = [synth_records[i] for i in approved_ids if i in synth_records]
    print(f"\n[eval] {total_rows} reviewed rows -> {len(approved_records)} approved")

    if not approved_records:
        print("Nothing approved yet — fill in the 'approve' column and re-run.")
        return

    existing = load_jsonl(EVAL_FILE)
    merged = existing + approved_records
    write_jsonl(EVAL_FILE, merged, dry_run)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Merge synthetic DebateAI data")
    sub = parser.add_subparsers(dest="command", required=True)

    p_rag = sub.add_parser("rag")
    p_rag.add_argument("--domain", choices=["gov", "worldbank"], required=True)
    p_rag.add_argument("--dry-run", action="store_true")

    p_fal = sub.add_parser("fallacies")
    p_fal.add_argument("--dry-run", action="store_true")

    p_ev_r = sub.add_parser("eval_review")
    p_ev_r.add_argument("--dry-run", action="store_true")

    p_ev_c = sub.add_parser("eval_commit")
    p_ev_c.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "rag":
        merge_rag(args.domain, args.dry_run)
    elif args.command == "fallacies":
        merge_fallacies(args.dry_run)
    elif args.command == "eval_review":
        eval_review(args.dry_run)
    elif args.command == "eval_commit":
        eval_commit(args.dry_run)


if __name__ == "__main__":
    main()
