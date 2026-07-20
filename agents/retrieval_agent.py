import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from agents.base_agent import BaseAgent
from agents.schemas import RetrievalInput, RetrievalOutput, ChunkMetadata

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
VDB_DIR = ROOT / "vector_db" / "faiss_index"
META_DB = ROOT / "vector_db" / "index_metadata.db"
RAG_DIR = ROOT / "data" / "processed" / "rag_chunks"

class RetrievalAgent(BaseAgent):
    """
    Retrieves context chunks for a factual claim using BGE embeddings + FAISS,
    followed by CrossEncoder reranking. Cache is preserved per-session in-memory.
    """

    def __init__(self):
        super().__init__()
        log.info("Initializing RetrievalAgent...")
        # 1. Load FAISS index
        import faiss
        idx_path = VDB_DIR / "index.faiss"
        if idx_path.exists():
            self.index = faiss.read_index(str(idx_path))
            log.info("  FAISS index loaded successfully.")
        else:
            log.warning("  FAISS index file not found at %s. Retrieval will be unavailable.", idx_path)
            self.index = None

        # 2. Load Embedding Model
        self.embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        log.info("  Embedding model BAAI/bge-large-en-v1.5 loaded.")

        # 3. Load Reranker Model (optional fallback)
        try:
            from sentence_transformers import CrossEncoder
            # Reranker is optional, if loading takes too long or fails we set to None
            log.info("  Attempting to load CrossEncoder reranker BAAI/bge-reranker-large...")
            # Set a timeout or catch error
            self.reranker = CrossEncoder("BAAI/bge-reranker-large")
            log.info("  CrossEncoder reranker loaded successfully.")
        except Exception as e:
            log.warning("  Could not load BAAI/bge-reranker-large reranker: %s. Using FAISS scores only.", e)
            self.reranker = None

        # 4. Load full texts of chunks from JSONL into memory
        self.chunk_texts = {}
        self._load_full_chunks()

        # 5. Session Cache
        self._cache = {}

    def _load_full_chunks(self):
        """Pre-loads all full chunk texts from JSONL files to enrich SQLite metadata previews."""
        if not RAG_DIR.exists():
            return
        for fpath in RAG_DIR.glob("*_chunks.jsonl"):
            try:
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            cid = data.get("chunk_id")
                            if cid:
                                self.chunk_texts[cid] = data.get("text", "")
            except Exception as e:
                log.error("Failed to load chunk file %s: %s", fpath, e)
        log.info("  Loaded %d full chunk texts into memory cache.", len(self.chunk_texts))

    def get_fallback_output(self, input: RetrievalInput, error_msg: str = "Timeout occurred") -> RetrievalOutput:
        return RetrievalOutput(claim=input.claim, chunks=[], error=error_msg)

    def _sha256(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def run(self, input: RetrievalInput) -> RetrievalOutput:
        claim = input.claim.strip()
        if not claim:
            return RetrievalOutput(claim=claim, chunks=[])

        # Session cache check
        chash = self._sha256(claim)
        if chash in self._cache:
            log.info("  Retrieval cache hit for claim: %r", claim[:50])
            return RetrievalOutput(claim=claim, chunks=self._cache[chash])

        if not self.index:
            return self.get_fallback_output(input, "FAISS index not loaded.")

        # 1. Embed query claim
        import numpy as np
        # BGE embeddings must be L2-normalized for Inner Product (cosine sim) search
        q_emb = self.embed_model.encode([claim], normalize_embeddings=True).astype(np.float32)

        # 2. Search FAISS index for top-8 candidates
        top_k = min(8, self.index.ntotal)
        if top_k == 0:
            return RetrievalOutput(claim=claim, chunks=[])

        scores, indices = self.index.search(q_emb, top_k)

        # 3. Retrieve metadata from SQLite
        candidates = []
        conn = sqlite3.connect(str(META_DB))
        cur = conn.cursor()

        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            cur.execute("""
                SELECT chunk_id, source_url, title, trust_tier, domain_topic 
                FROM chunk_metadata WHERE row_id = ?
            """, (idx,))
            row = cur.fetchone()
            if row:
                cid, url, title, tier, topic = row
                # Fetch full text from memory cache if available, else text_preview is a fallback
                full_text = self.chunk_texts.get(cid, "")
                if not full_text:
                    cur.execute("SELECT text_preview FROM chunk_metadata WHERE row_id = ?", (idx,))
                    preview = cur.fetchone()
                    full_text = preview[0] if preview else ""

                candidates.append(ChunkMetadata(
                    chunk_id=cid,
                    text=full_text,
                    source_url=url,
                    title=title,
                    trust_tier=tier,
                    domain_topic=topic,
                    score=float(score)
                ))
        conn.close()

        # 4. Rerank down to top 3-4
        if self.reranker and candidates:
            try:
                pairs = [[claim, c.text] for c in candidates]
                rerank_scores = self.reranker.predict(pairs)
                for c, r_score in zip(candidates, rerank_scores):
                    c.score = float(r_score)
                # Sort descending by rerank score
                candidates.sort(key=lambda x: x.score, reverse=True)
                final_chunks = candidates[:4]
                log.info("  Reranked %d candidates down to top %d.", len(candidates), len(final_chunks))
            except Exception as e:
                log.warning("  Reranking failed: %s. Falling back to FAISS scores.", e)
                candidates.sort(key=lambda x: x.score, reverse=True)
                final_chunks = candidates[:4]
        else:
            candidates.sort(key=lambda x: x.score, reverse=True)
            final_chunks = candidates[:4]

        # Populate cache
        self._cache[chash] = final_chunks
        return RetrievalOutput(claim=claim, chunks=final_chunks)

# Standalone verification runner
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    async def test():
        agent = RetrievalAgent()
        test_input = RetrievalInput(claim="WHO air pollution health effects")
        print("\nRunning standalone RetrievalAgent test...")
        output = await agent.run(test_input)
        print("Output:", output.model_dump_json(indent=2))

    asyncio.run(test())
