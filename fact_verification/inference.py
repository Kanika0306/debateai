import os
import sys
import asyncio
import logging
from pathlib import Path
import torch
from typing import Optional, List, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from fact_verification.config import CFG
except ImportError:
    from config import CFG

from agents.schemas import FactVerificationInput, FactVerificationOutput, ChunkMetadata

log = logging.getLogger(__name__)
CKPT = os.path.join(CFG.output_dir, "best")

LABEL_MAPPING = {
    "SUPPORTS": "True",
    "REFUTES": "False",
    "NOT_ENOUGH_INFO": "Unverified"
}


class LocalFactVerificationAgent:
    """
    Local DeBERTa-v3 model for fact verification (Model 2).
    Confidence-gated: predictions above threshold (default 0.70) are returned;
    predictions below threshold return None to trigger LLM fallback.
    """

    def __init__(self, checkpoint_dir: str = CKPT, device: str = None, threshold: float = 0.70):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        self.model.to(self.device).eval()
        self.id2label = self.model.config.id2label
        self.confidence_threshold = threshold

    def _predict_sync(self, claim: str, evidence_text: str):
        with torch.no_grad():
            enc = self.tokenizer(
                evidence_text, claim, truncation=True, max_length=CFG.max_length,
                padding=True, return_tensors="pt"
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
            conf, pred_id = probs.max(dim=-1)
            raw_label = self.id2label[pred_id.item()]
            mapped_verdict = LABEL_MAPPING.get(raw_label, "Unverified")
            return mapped_verdict, float(conf.item())

    async def verify(self, input_obj: FactVerificationInput) -> Optional[FactVerificationOutput]:
        """
        Verifies a FactVerificationInput claim against evidence using local model.
        Returns FactVerificationOutput if confidence >= threshold, else None.
        """
        evidence_text = "\n".join([c.text for c in input_obj.evidence]) if input_obj.evidence else ""
        verdict, confidence = await asyncio.to_thread(self._predict_sync, input_obj.claim, evidence_text)

        if confidence < self.confidence_threshold:
            return None  # Signal orchestrator to use LLM fallback

        cited_chunks = [c.chunk_id for c in input_obj.evidence] if input_obj.evidence and verdict in ("True", "False") else []
        return FactVerificationOutput(
            claim=input_obj.claim,
            verdict=verdict,
            confidence=confidence,
            cited_chunks=cited_chunks
        )


if __name__ == "__main__":
    async def _demo():
        logging.basicConfig(level=logging.INFO)
        agent = LocalFactVerificationAgent()
        sample_input = FactVerificationInput(
            claim="The Eiffel Tower is located in Paris.",
            evidence=[
                ChunkMetadata(
                    chunk_id="chunk_01",
                    text="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
                    source_url="https://example.com/paris",
                    title="Eiffel Tower",
                    trust_tier=1,
                    domain_topic="geography"
                )
            ]
        )
        result = await agent.verify(sample_input)
        print("Demo verify result:", result)

    asyncio.run(_demo())

