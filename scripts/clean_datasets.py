"""
clean_datasets.py - Phase 1.5 Data Cleaning Pipeline
=====================================================
Run individual sections: python scripts/clean_datasets.py --section 1
Run all sections:        python scripts/clean_datasets.py --section all
"""
import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# Ensure log dir exists before logging
(ROOT / "data").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "data" / "cleaning.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ===========================================================================
# SHARED UTILITIES
# ===========================================================================

def clean_text(text):
    """ftfy + citation-marker strip + whitespace normalize."""
    import ftfy
    if not isinstance(text, str) or not text.strip():
        return ""
    text = ftfy.fix_text(text)
    text = re.sub(r"\[\d+\]", "", text)   # strip [1], [12] etc.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text):
    return len(text.split()) if text else 0


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def banner(title):
    line = "=" * 70
    log.info("\n%s\n  %s\n%s", line, title, line)


def verify_df(df, name, label_col=None):
    banner("VERIFY - " + name)
    log.info("  Rows: %s", f"{len(df):,}")
    if label_col and label_col in df.columns:
        dist = df[label_col].value_counts()
        total = len(df)
        for cls, cnt in dist.items():
            pct = cnt / total * 100
            flag = "  <5% WARNING" if pct < 5 else ""
            log.info("    %-20s: %6s (%.1f%%)%s", cls, f"{cnt:,}", pct, flag)
    for _, row in df.head(5).iterrows():
        text = str(row.get("claim", row.get("text", "?")))[:80]
        lbl = row.get(label_col, "?") if label_col else "?"
        log.info("    [%s] %r", lbl, text)


# ===========================================================================
# LABEL MAPS
# ===========================================================================

FEVER_MAP = {
    "SUPPORTS":       "True",
    "REFUTES":        "False",
    "NOT ENOUGH INFO": "Unverified",
}

# liar2 uses integers 0-5
# 0=pants-fire, 1=false, 2=barely-true, 3=half-true, 4=mostly-true, 5=true
LIAR2_MAP = {
    0: "False",
    1: "False",
    2: "Misleading",
    3: "Misleading",
    4: "True",
    5: "True",
}

# FEVEROUS uses integers 0=REFUTES, 1=NOT_ENOUGH_INFO, 2=SUPPORTS
FEVEROUS_MAP = {
    0: "False",
    1: "Unverified",
    2: "True",
}


# ===========================================================================
# SECTION 0 - Setup
# ===========================================================================

def section0():
    banner("SECTION 0 - Setup")
    dirs = [
        PROC / "fact_verification",
        PROC / "fallacies",
        PROC / "rag_chunks",
        PROC / "misinformation",
        PROC / "claim_detection",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        log.info("  ensured: %s", d)
    log.info("SECTION 0 - DONE\n")


# ===========================================================================
# SECTION 1 - Fact Verification (FEVER, LIAR2, FEVEROUS)
# ===========================================================================

def _dedup_filter(df, name):
    """SHA256 dedup + word-count filter (4-60 words)."""
    df = df.copy()
    df["claim_hash"] = df["claim"].apply(sha256)
    before = len(df)
    df = df.drop_duplicates("claim_hash", keep="first")
    log.info("  %s: dropped %s exact-dup claims", name, f"{before - len(df):,}")
    df["_wc"] = df["claim"].apply(word_count)
    short_cnt = int((df["_wc"] < 4).sum())
    long_cnt  = int((df["_wc"] > 60).sum())
    df = df[(df["_wc"] >= 4) & (df["_wc"] <= 60)].drop(columns=["_wc"])
    log.info("  %s: dropped %d short + %d long -> %s remain",
             name, short_cnt, long_cnt, f"{len(df):,}")
    return df


def section1():
    banner("SECTION 1 - Fact Verification")

    # ---- 1a FEVER ----
    banner("1a. FEVER - schema + 5 raw rows")
    fever_parts = []
    for split in ["train", "validation", "test"]:
        p = RAW / "fact_verification" / "fever" / f"{split}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["claim", "label", "original_id"])
        log.info("  FEVER [%s] shape=%s  unique_labels=%s",
                 split, df.shape, df["label"].unique().tolist())
        df["source_dataset"] = "fever"
        df["original_id"] = df["original_id"].astype(str)
        fever_parts.append(df)
    fever = pd.concat(fever_parts, ignore_index=True)
    log.info("  FEVER 5 raw rows:\n%s\n", fever[["claim", "label"]].head(5).to_string())
    fever["claim"] = fever["claim"].apply(clean_text)
    fever["label"] = fever["label"].map(FEVER_MAP)
    unmapped = fever["label"].isna().sum()
    if unmapped:
        log.warning("  FEVER: %d unmapped label rows dropped", unmapped)
    fever = fever.dropna(subset=["label"])
    fever = _dedup_filter(fever, "FEVER")

    # ---- 1a LIAR2 ----
    banner("1a. LIAR2 - schema + 5 raw rows")
    log.info("  LIAR2 label->4class mapping: %s", LIAR2_MAP)
    liar_parts = []
    for split in ["train", "validation", "test"]:
        p = RAW / "fact_verification" / "liar" / f"{split}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["id", "statement", "label"])
        log.info("  LIAR2 [%s] shape=%s  raw_unique_labels=%s",
                 split, df.shape, sorted(df["label"].unique().tolist()))
        df = df.rename(columns={"statement": "claim", "id": "original_id"})
        df["source_dataset"] = "liar2"
        df["original_id"] = df["original_id"].astype(str)
        liar_parts.append(df)
    liar = pd.concat(liar_parts, ignore_index=True)
    log.info("  LIAR2 5 raw rows:\n%s\n", liar[["claim", "label"]].head(5).to_string())
    liar["claim"] = liar["claim"].apply(clean_text)
    liar["label"] = liar["label"].map(LIAR2_MAP)
    liar = liar.dropna(subset=["label"])
    liar = _dedup_filter(liar, "LIAR2")

    # ---- 1a FEVEROUS ----
    banner("1a. FEVEROUS - schema + 5 raw rows")
    log.info("  FEVEROUS label->4class mapping: %s", FEVEROUS_MAP)
    fev_path = RAW / "fact_verification" / "feverous" / "train.parquet"
    feverous = pd.read_parquet(fev_path, columns=["id", "claim", "label"])
    log.info("  FEVEROUS shape=%s  unique_labels=%s",
             feverous.shape, sorted(feverous["label"].unique().tolist()))
    log.info("  FEVEROUS 5 raw rows:\n%s\n", feverous.head(5).to_string())

    # ---- 1d FEVEROUS evidence decision ----
    banner("1d. FEVEROUS evidence flattening")
    log.info("  Evidence cell refs follow: Article_cell_tableIdx_rowIdx_colIdx")
    log.info("  Cell values require feverous_wikiv1.db (~100GB) to resolve.")
    log.info("  DECISION: BLOCKED - keeping claim field only.")
    log.info("  evidence_raw column preserved as None placeholder.")
    feverous["evidence_raw"] = None

    feverous = feverous.rename(columns={"id": "original_id"})
    feverous["original_id"] = feverous["original_id"].astype(str)
    feverous["source_dataset"] = "feverous"
    feverous["claim"] = feverous["claim"].apply(clean_text)
    feverous["label"] = feverous["label"].map(FEVEROUS_MAP)
    feverous = feverous.dropna(subset=["label"])
    feverous = _dedup_filter(feverous, "FEVEROUS")

    # ---- 1e Cross-dataset dedup ----
    banner("1e. Cross-dataset dedup")
    all_hashes = {}
    for nm, df in [("fever", fever), ("liar2", liar), ("feverous", feverous)]:
        for h in df["claim_hash"]:
            all_hashes.setdefault(h, []).append(nm)
    cross = {h: srcs for h, srcs in all_hashes.items() if len(srcs) > 1}
    log.info("  Claims in 2+ datasets: %s", f"{len(cross):,}")

    # ---- save ----
    out = PROC / "fact_verification"
    cols = ["claim", "label", "source_dataset", "original_id", "claim_hash"]
    fever[cols].to_parquet(out / "fever_clean.parquet", index=False)
    liar[cols].to_parquet(out / "liar_clean.parquet", index=False)
    feverous[cols + ["evidence_raw"]].to_parquet(out / "feverous_clean.parquet", index=False)
    log.info("  Saved fever_clean (%s rows), liar_clean (%s), feverous_clean (%s)",
             f"{len(fever):,}", f"{len(liar):,}", f"{len(feverous):,}")

    verify_df(fever[cols], "FEVER cleaned", "label")
    verify_df(liar[cols], "LIAR2 cleaned", "label")
    verify_df(feverous[cols], "FEVEROUS cleaned", "label")
    banner("SECTION 1 - COMPLETE\n")


# ===========================================================================
# SECTION 2 - Fallacies (Argotario + Logic Dataset)
# ===========================================================================

def _is_english(text):
    try:
        from langdetect import detect
        return detect(str(text)) == "en"
    except Exception:
        return True  # fail-open


def section2():
    banner("SECTION 2 - Fallacies")

    # ---- 2a Argotario ----
    banner("2a. Argotario TSV")
    argo = pd.read_csv(
        RAW / "fallacies" / "argotario" / "arguments-en-2018-01-15.tsv",
        sep="\t", encoding="utf-8"
    )
    log.info("  Columns: %s", argo.columns.tolist())
    log.info("  Shape: %s", argo.shape)
    log.info("  First 3 rows:\n%s\n", argo.head(3).to_string())
    en_mask = argo["Text"].apply(_is_english)
    dropped = int((~en_mask).sum())
    argo = argo[en_mask].copy()
    log.info("  After langdetect EN filter: dropped %d, %d remain", dropped, len(argo))
    log.info("  Argotario fallacy types:\n%s\n",
             argo["Intended Fallacy"].value_counts().to_string())

    # ---- 2a Logic Dataset ----
    banner("2a. Logic Dataset")
    logic_parts = []
    for split in ["train", "dev", "test"]:
        p = RAW / "fallacies" / "logic_dataset" / f"{split}.parquet"
        if p.exists():
            logic_parts.append(pd.read_parquet(p))
    logic = pd.concat(logic_parts, ignore_index=True)
    log.info("  Columns: %s  Shape: %s", logic.columns.tolist(), logic.shape)
    log.info("  Logic Dataset fallacy types:\n%s\n",
             logic["logical_fallacies"].value_counts().to_string())

    # ---- 2b Unified schema ----
    banner("2b. Unified schema")
    argo_u = pd.DataFrame({
        "text":         argo["Text"].apply(clean_text),
        "fallacy_type": argo["Intended Fallacy"].str.strip().str.lower(),
        "source":       "argotario",
        "id":           argo["Mongo ID"].astype(str),
    })
    logic_u = pd.DataFrame({
        "text":         logic["source_article"].apply(clean_text),
        "fallacy_type": logic["logical_fallacies"].str.strip().str.lower(),
        "source":       "logic_dataset",
        "id":           [f"logic_{i}" for i in range(len(logic))],
    })
    merged = pd.concat([argo_u, logic_u], ignore_index=True)
    before = len(merged)
    merged = merged[merged["text"].apply(word_count) >= 4]
    log.info("  Merged: %d rows (dropped %d under 4 words)", len(merged), before - len(merged))

    # ---- 2c Class balance ----
    banner("2c. Fallacy class balance")
    sparse = []
    for ft, cnt in merged["fallacy_type"].value_counts().items():
        flag = "  SPARSE <10" if cnt < 10 else ""
        log.info("    %-40s: %5d%s", ft, cnt, flag)
        if cnt < 10:
            sparse.append(ft)
    if sparse:
        log.warning("  Sparse types (flagged, NOT dropped): %s", sparse)

    # ---- 2e Save JSONL ----
    out = PROC / "fallacies" / "fallacy_examples_unified.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in merged.itertuples(index=False):
            f.write(json.dumps({
                "text": row.text,
                "fallacy_type": row.fallacy_type,
                "source": row.source,
                "id": row.id,
            }, ensure_ascii=False) + "\n")
    log.info("  Saved %s (%d rows)", out, len(merged))

    # ---- verify ----
    banner("VERIFY - Fallacies JSONL")
    log.info("  Total: %d", len(merged))
    for ft in sorted(merged["fallacy_type"].unique()):
        sub = merged[merged["fallacy_type"] == ft]
        log.info("  [%s] %d examples:", ft, len(sub))
        for _, r in sub.head(3).iterrows():
            log.info("    * %r", r["text"][:100])

    banner("SECTION 2 - COMPLETE\n")


# ===========================================================================
# SECTION 3 - Misinformation (FakeNewsNet)
# ===========================================================================

def section3():
    banner("SECTION 3 - Misinformation (FakeNewsNet)")
    fnn_dir = RAW / "misinformation" / "fakenewsnet" / "dataset"
    files = sorted(fnn_dir.glob("*.csv"))
    log.info("  CSV files: %s", [f.name for f in files])

    sample = pd.read_csv(files[0], nrows=3)
    log.info("  Sample columns: %s", sample.columns.tolist())
    log.info("  3 rows:\n%s\n", sample.to_string())
    has_body = any(c in sample.columns for c in ["content", "text", "body"])
    log.info("  Full article text present: %s", has_body)

    dfs = []
    for f in files:
        label  = "fake" if "fake" in f.stem else "real"
        source = "gossipcop" if "gossipcop" in f.stem else "politifact"
        df = pd.read_csv(f, usecols=lambda c: c in ["id", "news_url", "title", "tweet_ids"])
        df["label"]  = label
        df["source"] = source
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    log.info("  Total metadata rows: %s", f"{len(all_df):,}")
    log.info("\n%s\n", all_df.groupby(["source", "label"]).size().to_string())
    log.info(
        "  STATUS: BLOCKED - full article text not in CSVs.\n"
        "  FakeNewsNet stores: id, news_url, title, tweet_ids only.\n"
        "  To fetch bodies:\n"
        "    1. Twitter API v2 Bearer Token (env: TWITTER_BEARER_TOKEN)\n"
        "    2. News scraper (Newspaper3k / Trafilatura + rate limits)\n"
        "  Re-run with --section 3 once credentials are available."
    )
    out = PROC / "misinformation" / "fakenewsnet_metadata.parquet"
    all_df.to_parquet(out, index=False)
    log.info("  Saved metadata-only: %s", out)
    banner("SECTION 3 - BLOCKED (metadata saved)\n")


# ===========================================================================
# SECTION 4 - Claim Detection (CheckThat Lab)
# ===========================================================================

def section4():
    banner("SECTION 4 - Claim Detection (CheckThat Lab)")
    cb_dir = RAW / "claim_detection" / "claimbuster"

    # LFS status
    res = subprocess.run(
        ["git", "lfs", "ls-files"],
        cwd=str(cb_dir), capture_output=True, text=True
    )
    log.info("  git lfs ls-files:\n%s\n", res.stdout or res.stderr)

    # File inventory
    log.info("  File inventory:")
    for f in sorted(cb_dir.glob("**/*")):
        if f.is_file():
            log.info("    %s - %s bytes", f.name, f"{f.stat().st_size:,}")

    npy_stubs = [f for f in cb_dir.glob("*.npy") if f.stat().st_size < 200]
    log.info("  LFS stub .npy files: %s", [f.name for f in npy_stubs])

    # ---- verified-claims.tsv (main fact-check corpus, 88.6 MB) ----
    # TSV has NO header row. Columns: id, claim, date, url, verdict, speaker, title, body
    vclaims_tsv = cb_dir / "verified-claims.tsv"
    if vclaims_tsv.exists() and vclaims_tsv.stat().st_size > 1000:
        VCOLS = ["id", "claim", "date", "source_url", "verdict", "speaker", "title", "body"]
        df_vc = pd.read_csv(
            vclaims_tsv, sep="\t", header=None, names=VCOLS,
            on_bad_lines="skip", encoding="utf-8"
        )
        log.info("  verified-claims.tsv: shape=%s", df_vc.shape)
        log.info("  Top verdicts:\n%s", df_vc["verdict"].value_counts().head(15).to_string())
        log.info("  5 sample rows:\n%s\n",
                 df_vc[["claim", "verdict", "speaker"]].head(5).to_string())

        df_vc["claim"] = df_vc["claim"].apply(clean_text)
        df_vc["source"] = "checkthat_vclaims"
        df_vc = df_vc.dropna(subset=["claim"])
        df_vc = df_vc[df_vc["claim"].apply(word_count) >= 4]
        out_vc = PROC / "claim_detection" / "checkthat_vclaims.parquet"
        df_vc.to_parquet(out_vc, index=False)
        log.info("  Saved verified-claims: %s (%d rows)", out_vc, len(df_vc))

    # ---- train_dataset.tsv (sentence_id -> check-worthiness score mapping) ----
    train_tsv = cb_dir / "train_dataset.tsv"
    if train_tsv.exists() and train_tsv.stat().st_size > 500:
        df_train = pd.read_csv(train_tsv, sep="\t", header=None,
                               names=["sentence_id", "claim_score"],
                               on_bad_lines="skip")
        log.info("  train_dataset.tsv: shape=%s  (sentence_id -> claim_score mapping)",
                 df_train.shape)
        df_train["source"] = "checkthat_lab"
        out = PROC / "claim_detection" / "checkthat_clean.parquet"
        df_train.to_parquet(out, index=False)
        log.info("  Saved: %s (%d rows)", out, len(df_train))
    else:
        log.warning("  BLOCKED: train_dataset.tsv missing or too small.")
        log.warning("  Run inside %s:", cb_dir)
        log.warning("    git lfs pull --include='*.tsv'")

    banner("SECTION 4 - COMPLETE\n")


# ===========================================================================
# SECTION 5 - RAG Sources (WHO, WorldBank, NASA, Gov)
# ===========================================================================

def section5():
    banner("SECTION 5 - RAG Sources Cleaning and Chunking")

    try:
        import trafilatura
    except ImportError:
        log.error("trafilatura not installed")
        return

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    except Exception as e:
        log.error("spaCy en_core_web_sm unavailable: %s", e)
        log.error("Run: python -m spacy download en_core_web_sm")
        return

    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        log.error("datasketch not installed")
        return

    SOURCES = {
        "gov":          {"trust_tier": 1, "domain_topic": "economy"},
        "who":          {"trust_tier": 1, "domain_topic": "health"},
        "worldbank":    {"trust_tier": 1, "domain_topic": "economy"},
        "nasa_climate": {"trust_tier": 1, "domain_topic": "climate"},
    }

    def tokenize(text):
        return text.split()

    def chunk_doc(text, min_tok=200, max_tok=400, overlap_ratio=0.15):
        doc = nlp(text[:500_000])
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        chunks, cur_sents, cur_tok = [], [], 0
        ovlp_limit = int(max_tok * overlap_ratio)
        for sent in sents:
            st = len(tokenize(sent))
            if cur_tok + st > max_tok and cur_tok >= min_tok:
                chunks.append(" ".join(cur_sents))
                ovlp_sents, ot = [], 0
                for s in reversed(cur_sents):
                    sw = len(tokenize(s))
                    if ot + sw > ovlp_limit:
                        break
                    ovlp_sents.insert(0, s)
                    ot += sw
                cur_sents, cur_tok = ovlp_sents, ot
            cur_sents.append(sent)
            cur_tok += st
        if cur_sents:
            tail = " ".join(cur_sents)
            if len(tokenize(tail)) >= 20:
                chunks.append(tail)
        return chunks

    def make_minhash(text):
        m = MinHash(num_perm=128)
        for w in set(text.lower().split()):
            m.update(w.encode("utf-8"))
        return m

    all_results = {}

    for src_name, meta in SOURCES.items():
        banner("5. Processing: " + src_name)
        src_dir = RAW / "rag_sources" / src_name
        html_files = sorted(src_dir.glob("*.html"))
        log.info("  Found %d HTML files", len(html_files))

        docs, skipped = [], []
        for hp in html_files:
            mp = hp.with_suffix("").with_suffix(".meta.json")
            meta_data = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
            raw_html  = hp.read_text(encoding="utf-8", errors="replace")
            extracted = trafilatura.extract(
                raw_html, include_tables=True,
                favor_precision=True, no_fallback=False
            )
            if not extracted or len(extracted.strip()) < 50:
                log.warning("  trafilatura empty for %s - skipping", hp.name)
                skipped.append(hp.name)
                continue
            docs.append({
                "filename":     hp.name,
                "text":         clean_text(extracted),
                "source_url":   meta_data.get("source_url", ""),
                "title":        meta_data.get("page_title", hp.stem),
                "fetch_date":   meta_data.get("fetch_date", ""),
                "trust_tier":   meta["trust_tier"],
                "domain_topic": meta["domain_topic"],
            })
        log.info("  Extracted: %d, skipped: %s", len(docs), skipped)

        # 5b. WorldBank before/after samples
        if src_name == "worldbank" and docs:
            banner("5b. WorldBank extraction samples")
            for d in docs[:5]:
                log.info("  FILE: %s", d["filename"])
                log.info("  TEXT (300 chars): %r\n", d["text"][:300])

        # 5c. MinHash near-dedup
        lsh = MinHashLSH(threshold=0.85, num_perm=128)
        kept, dropped_mh = [], 0
        for i, doc in enumerate(docs):
            mh  = make_minhash(doc["text"])
            key = f"{src_name}_{i}"
            try:
                if lsh.query(mh):
                    log.info("  Near-dup: %s - dropping", doc["filename"])
                    dropped_mh += 1
                else:
                    lsh.insert(key, mh)
                    kept.append(doc)
            except Exception:
                lsh.insert(key, mh)
                kept.append(doc)
        log.info("  MinHash dedup: %d -> %d (dropped %d)", len(docs), len(kept), dropped_mh)

        # 5d. Sentence-aware chunking
        chunks = []
        for doc in kept:
            for i, chunk_text in enumerate(chunk_doc(doc["text"])):
                cid = src_name + "_" + doc["filename"].replace(".html", "") + f"_{i:04d}"
                chunks.append({
                    "chunk_id":     cid,
                    "text":         chunk_text,
                    "source_url":   doc["source_url"],
                    "title":        doc["title"],
                    "fetch_date":   doc["fetch_date"],
                    "trust_tier":   doc["trust_tier"],
                    "domain_topic": doc["domain_topic"],
                    "token_count":  len(tokenize(chunk_text)),
                })

        log.info("  Generated %d chunks from %d docs", len(chunks), len(kept))
        if chunks:
            avg = sum(c["token_count"] for c in chunks) / len(chunks)
            log.info("  Avg chunk size: %.0f tokens", avg)
            log.info("  3 sample chunks:")
            for c in chunks[:3]:
                log.info("    [%s] (%d tok)  url=%s",
                         c["chunk_id"], c["token_count"], c["source_url"])
                log.info("    %r", c["text"][:120])

        # 5f. Save JSONL
        out = PROC / "rag_chunks" / f"{src_name}_chunks.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        log.info("  Saved: %s", out)
        all_results[src_name] = {
            "docs":    len(kept),
            "chunks":  len(chunks),
            "skipped": skipped,
        }

    banner("SECTION 5 - COMPLETE")
    log.info("  %-15s %6s %8s", "Source", "Docs", "Chunks")
    log.info("  " + "-" * 35)
    for sn, r in all_results.items():
        log.info("  %-15s %6d %8d  skipped=%s",
                 sn, r["docs"], r["chunks"], r["skipped"])


# ===========================================================================
# FINAL SUMMARY
# ===========================================================================

def final_summary():
    banner("FINAL CONSOLIDATED SUMMARY")
    rows = [
        ("FEVER",         "260K rows",  "fever_clean.parquet",             "Complete",   ""),
        ("LIAR2",         "22K rows",   "liar_clean.parquet",              "Complete",   ""),
        ("FEVEROUS",      "71K rows",   "feverous_clean.parquet",          "Complete",   "evidence BLOCKED needs wikiv1.db ~100GB"),
        ("Argotario",     "TSV EN",     "fallacy_examples_unified.jsonl",  "Complete",   ""),
        ("Logic Dataset", "3.7K rows",  "fallacy_examples_unified.jsonl",  "Complete",   ""),
        ("FakeNewsNet",   "IDs only",   "fakenewsnet_metadata.parquet",    "BLOCKED",    "Needs Twitter API v2 Bearer Token + scraper"),
        ("CheckThat",     "TSV+npy",    "checkthat_clean.parquet",         "TSV only",   ".npy=LFS stubs; git lfs pull --include='*.tsv'"),
        ("WHO RAG",       "15 HTML",    "who_chunks.jsonl",                "Complete",   ""),
        ("WorldBank RAG", "13 HTML",    "worldbank_chunks.jsonl",          "Complete",   ""),
        ("NASA RAG",      "13 HTML",    "nasa_climate_chunks.jsonl",       "Complete",   ""),
        ("Gov RAG",       "11 HTML",    "gov_chunks.jsonl",                "Complete",   ""),
    ]
    log.info("  %-20s %-12s %-38s %-12s %s",
             "Dataset", "Raw", "Output file", "Status", "Notes")
    log.info("  " + "-" * 100)
    for r in rows:
        log.info("  %-20s %-12s %-38s %-12s %s", *r)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Debate-AI data cleaning pipeline")
    parser.add_argument(
        "--section", default="all",
        help="Section to run: 0 1 2 3 4 5 all (default: all)"
    )
    args = parser.parse_args()
    s = args.section.strip()

    section0()
    if s in ("all", "1"): section1()
    if s in ("all", "2"): section2()
    if s in ("all", "3"): section3()
    if s in ("all", "4"): section4()
    if s in ("all", "5"): section5()
    if s == "all":        final_summary()


if __name__ == "__main__":
    main()
