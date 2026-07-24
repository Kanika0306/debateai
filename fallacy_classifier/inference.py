"""
Inference wrapper for the trained fallacy classifier, shaped to match
your existing FallacyAgent contract (segment text in -> flagged spans +
taxonomy classification out) so it can be dropped in as a drop-in
replacement for, or ensemble partner to, the LLM-prompted FallacyAgent.

Example (async, matching your orchestrator's agent interface):

    from inference import LocalFallacyAgent
    agent = LocalFallacyAgent()
    results = await agent.analyze(segment_text)
    # -> [{"text": "...", "fallacy_type": "ad hominem", "confidence": 0.87}, ...]

Falls back gracefully: if a segment has no fallacy above `threshold`,
returns an empty list (mirrors "no fallacy found" behavior of the
LLM agent, rather than always forcing a label).
"""
import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import FINAL_MODEL_DIR, MAX_LENGTH


@dataclass
class FallacyFlag:
    text: str
    fallacy_type: str
    confidence: float

    def to_dict(self) -> dict:
        return {"text": self.text, "fallacy_type": self.fallacy_type, "confidence": self.confidence}


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class LocalFallacyAgent:
    """
    Drop-in local replacement for the GPT-4o-mini/Gemini FallacyAgent.
    Classifies at sentence granularity (matches how "flagged text span"
    is expected downstream) and only reports fallacies above `threshold`.
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = FINAL_MODEL_DIR,
        threshold: Union[float, dict] = 0.55,
        device: str = None,
    ):
        """
        threshold: either a single float applied to every class, or a dict
        mapping fallacy_type -> float for per-class cutoffs (recommended —
        see tune_threshold.py, which sweeps per-class thresholds on the val
        set and will usually tell you weak classes need a higher bar than
        strong ones). Classes missing from the dict fall back to
        `default_threshold`.

        Example per-class usage:
            agent = LocalFallacyAgent(threshold={
                "no fallacy": 0.45,
                "equivocation": 0.75,   # weak class -> higher bar
                "_default": 0.65,
            })
        """
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_dir}. Run train.py first, "
                f"or pass model_dir= explicitly."
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

        if isinstance(threshold, dict):
            self.default_threshold = threshold.get("_default", 0.55)
            self.per_class_threshold = threshold
        else:
            self.default_threshold = threshold
            self.per_class_threshold = {}

    def _threshold_for(self, label: str) -> float:
        return self.per_class_threshold.get(label, self.default_threshold)

    @torch.no_grad()
    def _classify_batch(self, sentences: List[str]):
        enc = self.tokenizer(
            sentences, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        confidences, pred_ids = torch.max(probs, dim=-1)
        return pred_ids.cpu().tolist(), confidences.cpu().tolist()

    def analyze_sync(self, segment_text: str) -> List[dict]:
        sentences = _split_sentences(segment_text)
        if not sentences:
            return []

        pred_ids, confidences = self._classify_batch(sentences)

        flags = []
        for sent, pred_id, conf in zip(sentences, pred_ids, confidences):
            label = self.id2label[pred_id]
            if label == "no fallacy":
                continue
            if conf < self._threshold_for(label):
                continue
            flags.append(FallacyFlag(text=sent, fallacy_type=label, confidence=round(conf, 4)))

        return [f.to_dict() for f in flags]

    async def analyze(self, segment_text: str) -> List[dict]:
        """Async wrapper — runs the (CPU/GPU-bound) forward pass in a thread
        so it doesn't block the event loop, matching your other async agents."""
        return await asyncio.to_thread(self.analyze_sync, segment_text)


if __name__ == "__main__":
    import sys

    agent = LocalFallacyAgent()
    text = " ".join(sys.argv[1:]) or (
        "You can't trust his climate policy, he's a college dropout. "
        "Everyone I know agrees renewable energy is the only answer."
    )
    print(f"Input: {text}\n")
    result = asyncio.run(agent.analyze(text))
    if not result:
        print("No fallacies detected.")
    else:
        for r in result:
            print(f"[{r['fallacy_type']} | conf={r['confidence']}] {r['text']}")
