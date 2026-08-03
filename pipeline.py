"""
pipeline.py
===========
Unified DebateAI Inference Pipeline.
Loads Claim Veracity Detector (v3), Fallacy Classifier (v4), and NLI Fact Verifier.
Executes un-gated analysis across all models with strict local path enforcement & input validation.
"""

import os
import time
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint paths
CLAIM_DETECTOR_PATH = os.path.abspath("./checkpoints/claim_detector_v3/best")
FALLACY_CLASSIFIER_PATH = os.path.abspath("./checkpoints/fallacy_classifier_v3/best")
NLI_MODEL_PATH = os.path.abspath("./checkpoints/nli_model/best")

print(f"[pipeline] Loading models on {DEVICE}...")

# Check local paths exist explicitly before loading
for name, path in [
    ("Claim Veracity (v3)", CLAIM_DETECTOR_PATH),
    ("Fallacy Classifier (v4)", FALLACY_CLASSIFIER_PATH),
    ("NLI Model", NLI_MODEL_PATH),
]:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"[pipeline ERROR] Local checkpoint directory does not exist for {name}: {path}"
        )

# Load tokenizers & models with local_files_only=True to prevent silent HF Hub fallbacks
veracity_tok = AutoTokenizer.from_pretrained(CLAIM_DETECTOR_PATH, local_files_only=True)
veracity_model = AutoModelForSequenceClassification.from_pretrained(CLAIM_DETECTOR_PATH, local_files_only=True).to(DEVICE).eval()

fallacy_tok = AutoTokenizer.from_pretrained(FALLACY_CLASSIFIER_PATH, local_files_only=True)
fallacy_model = AutoModelForSequenceClassification.from_pretrained(FALLACY_CLASSIFIER_PATH, local_files_only=True).to(DEVICE).eval()

nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL_PATH, local_files_only=True)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, local_files_only=True).to(DEVICE).eval()

# Load ID maps
with open(os.path.join(CLAIM_DETECTOR_PATH, "eval_report.json"), "r", encoding="utf-8") as f:
    veracity_id2label = json.load(f)["id2label"]
    veracity_id2label = {int(k): v for k, v in veracity_id2label.items()}

with open(os.path.join(FALLACY_CLASSIFIER_PATH, "eval_report.json"), "r", encoding="utf-8") as f:
    fallacy_id2label = json.load(f)["id2label"]
    fallacy_id2label = {int(k): v for k, v in fallacy_id2label.items()}

with open(os.path.join(NLI_MODEL_PATH, "eval_report.json"), "r", encoding="utf-8") as f:
    nli_report = json.load(f)
    nli_label_map = nli_report.get("label_map", {"NEI": 0, "REFUTES": 1, "SUPPORTS": 2})
    nli_id2label = {v: k for k, v in nli_label_map.items()}

print("[pipeline] All models loaded successfully!")


def analyze_text(text: str, evidence: str = None) -> dict:
    """Analyze a single text and optional evidence passage."""
    res = analyze_batch([text], [evidence] if evidence else None)
    return res[0]


def analyze_batch(texts: list[str], evidence: list[str] = None) -> list[dict]:
    """
    Batched inference pipeline for maximum GPU efficiency.
    Model forward passes are genuinely batched at model level.
    Runs veracity, fallacy, and optional NLI unconditionally without gates.
    """
    if not isinstance(texts, list):
        raise TypeError(f"texts argument must be a list of strings, got {type(texts)}")

    if not texts:
        return []

    # Validate each input text
    validated_texts = []
    for idx, t in enumerate(texts):
        if t is None or not isinstance(t, str) or not t.strip():
            raise ValueError(f"Input text at index {idx} must be a non-empty string. Got: {repr(t)}")
        validated_texts.append(t.strip())

    if evidence is not None:
        if not isinstance(evidence, list):
            raise TypeError(f"evidence argument must be a list if provided, got {type(evidence)}")
        if len(texts) != len(evidence):
            raise ValueError(
                f"texts and evidence lists must have equal length (got {len(texts)} texts and {len(evidence)} evidence items)"
            )
        evidence_list = evidence
    else:
        evidence_list = [None] * len(texts)

    # Step 1: Batched Veracity Classification (Claim Detector v3)
    veracity_inputs = veracity_tok(
        validated_texts,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        veracity_logits = veracity_model(**veracity_inputs).logits
        veracity_probs = F.softmax(veracity_logits, dim=-1).cpu().numpy()

    # Step 2: Batched Fallacy Classification (unconditional for all inputs)
    fallacy_inputs = fallacy_tok(
        validated_texts,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        fallacy_logits = fallacy_model(**fallacy_inputs).logits
        fallacy_probs = F.softmax(fallacy_logits, dim=-1).cpu().numpy()

    # Step 3: Batched NLI Fact Checking (unconditional for all inputs with evidence)
    nli_indices = []
    nli_pairs = []
    for i, ev in enumerate(evidence_list):
        if ev and isinstance(ev, str) and ev.strip():
            nli_indices.append(i)
            nli_pairs.append((validated_texts[i], ev.strip()))

    nli_results_map = {}
    if nli_indices:
        claims_batch = [pair[0] for pair in nli_pairs]
        evidences_batch = [pair[1] for pair in nli_pairs]

        nli_inputs = nli_tok(
            claims_batch,
            evidences_batch,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            nli_logits = nli_model(**nli_inputs).logits
            nli_probs_batch = F.softmax(nli_logits, dim=-1).cpu().numpy()

        for k, orig_idx in enumerate(nli_indices):
            probs_k = nli_probs_batch[k]
            top_id = probs_k.argmax()
            nli_results_map[orig_idx] = {
                "verdict": nli_id2label[int(top_id)],
                "confidence": round(float(probs_k[top_id]), 4)
            }

    # Assemble JSON responses according to exact schema
    results = []
    for i, text in enumerate(validated_texts):
        # Veracity prediction
        # NOTE (Known Domain Gap Limitation): Claim Detector v3 was trained on political fact-checks (LIAR/CheckThat).
        # Scientific or out-of-domain factual claims (e.g. "The Earth is 4.5 billion years old") may receive incorrect high confidence.
        v_probs = veracity_probs[i]
        v_top_id = v_probs.argmax()
        v_label = veracity_id2label[int(v_top_id)]
        v_conf = float(v_probs[v_top_id])
        v_reliable = bool(v_conf >= 0.60)

        # Fallacy prediction
        f_probs = fallacy_probs[i]
        f_top3_ids = f_probs.argsort()[-3:][::-1]
        f_top_id = f_top3_ids[0]
        f_top_label = fallacy_id2label[int(f_top_id)]
        f_top_conf = float(f_probs[f_top_id])

        f_top3 = [
            {
                "label": fallacy_id2label[int(idx)],
                "confidence": round(float(f_probs[idx]), 4)
            }
            for idx in f_top3_ids
        ]

        res = {
            "text": text,
            "veracity": {
                "label": v_label,
                "confidence": round(v_conf, 4),
                "reliable": v_reliable
            },
            "fallacy": {
                "label": f_top_label,
                "confidence": round(f_top_conf, 4),
                "top3": f_top3
            },
            "fact_check": nli_results_map.get(i, None),
            "model_versions": {
                "claim_detector": "v3",
                "fallacy_classifier": "v4",
                "nli_model": "roberta-large-mnli"
            }
        }
        results.append(res)

    return results
