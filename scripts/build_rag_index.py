"""
build_rag_index.py - Section 6: Build FAISS vector index over RAG chunks
=========================================================================
Run ONLY after Section 5 RAG chunks are verified.

Usage:
  python scripts/build_rag_index.py
  python scripts/build_rag_index.py --query "India GDP growth rate"

Install if needed:
  pip install sentence-transformers faiss-cpu
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
VDB  = ROOT / "vector_db" / "faiss_index"
META_DB = ROOT / "vector_db" / "index_metadata.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

CHUNK_SOURCES = [
    "gov_chunks.jsonl",
    "who_chunks.jsonl",
    "worldbank_chunks.jsonl",
    "nasa_climate_chunks.jsonl",
]

MODEL_NAME = "BAAI/bge-large-en-v1.5"

TEST_QUERIES = [
    "India GDP growth rate",
    "WHO air pollution health effects",
    "NASA global temperature rise",
]


def load_chunks():
    chunks = []
    for fname in CHUNK_SOURCES:
        path = PROC / "rag_chunks" / fname
        if not path.exists():
            log.warning("  Chunk file not found: %s", path)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    log.info("  Loaded %s total chunks from %d sources", f"{len(chunks):,}", len(CHUNK_SOURCES))
    return chunks


def build_index(chunks):
    try:
        import faiss
    except ImportError:
        log.error("faiss-cpu not installed. Run: pip install faiss-cpu")
        return None, None

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        return None, None

    log.info("  Loading embedding model: %s ...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    log.info("  Embedding %s chunks (this may take a few minutes on CPU)...", f"{len(texts):,}")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalize for cosine sim via inner product
    )
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    log.info("  Embedding dim: %d", dim)

    # IndexFlatIP = exact cosine similarity (since embeddings are L2-normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("  FAISS index built: %d vectors", index.ntotal)

    return index, model


def save_index(index, chunks):
    import faiss
    VDB.mkdir(parents=True, exist_ok=True)
    idx_path = VDB / "index.faiss"
    faiss.write_index(index, str(idx_path))
    log.info("  Saved FAISS index: %s", idx_path)

    # Save metadata to SQLite
    META_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(META_DB))
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS chunk_metadata")
    cur.execute("""
        CREATE TABLE chunk_metadata (
            row_id       INTEGER PRIMARY KEY,
            chunk_id     TEXT,
            source_url   TEXT,
            title        TEXT,
            fetch_date   TEXT,
            trust_tier   INTEGER,
            domain_topic TEXT,
            token_count  INTEGER,
            text_preview TEXT
        )
    """)
    rows = []
    for i, c in enumerate(chunks):
        rows.append((
            i,
            c.get("chunk_id", ""),
            c.get("source_url", ""),
            c.get("title", ""),
            c.get("fetch_date", ""),
            c.get("trust_tier", 1),
            c.get("domain_topic", ""),
            c.get("token_count", 0),
            c.get("text", "")[:200],
        ))
    cur.executemany("INSERT INTO chunk_metadata VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    log.info("  Saved metadata SQLite: %s (%d rows)", META_DB, len(chunks))


def run_test_queries(index, model, chunks):
    log.info("\n" + "=" * 70)
    log.info("  TEST QUERIES")
    log.info("=" * 70)
    for query in TEST_QUERIES:
        log.info("\n  Query: %r", query)
        q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = index.search(q_emb, 3)
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            c = chunks[idx]
            log.info(
                "    [Rank %d] score=%.4f  chunk_id=%s",
                rank, score, c.get("chunk_id", "?")
            )
            log.info("      url:   %s", c.get("source_url", ""))
            log.info("      title: %s", c.get("title", ""))
            log.info("      text:  %r", c.get("text", "")[:150])


def main():
    parser = argparse.ArgumentParser(description="Build FAISS RAG vector index")
    parser.add_argument("--query", default=None,
                        help="Run a single test query after building the index")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("  SECTION 6 - Build Vector Index")
    log.info("=" * 70)

    # 6a-b: Load chunks + embed
    chunks = load_chunks()
    if not chunks:
        log.error("No chunks found. Run: python scripts/clean_datasets.py --section 5 first.")
        sys.exit(1)

    index, model = build_index(chunks)
    if index is None:
        log.error("Index build failed. Check missing dependencies above.")
        sys.exit(1)

    # 6d: Save index + metadata DB
    save_index(index, chunks)

    # 6e: Test queries
    if args.query:
        TEST_QUERIES.insert(0, args.query)
    run_test_queries(index, model, chunks)

    log.info("\n  SECTION 6 - COMPLETE")
    log.info("  Index: %s", VDB / "index.faiss")
    log.info("  Metadata DB: %s", META_DB)


if __name__ == "__main__":
    main()
