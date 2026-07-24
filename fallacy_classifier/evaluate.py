"""
Standalone evaluation of a saved fallacy classifier checkpoint.
Prints a full classification report + confusion matrix, and saves both
to disk next to the model for your records.

Usage:
    python evaluate.py --model-dir models/fallacy-classifier-v1 --split test
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import LABELS, SPLITS_DIR, FINAL_MODEL_DIR, MAX_LENGTH

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def run_eval(model_dir: Path, split: str, batch_size: int = 32):
    split_path = SPLITS_DIR / f"{split}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}. Run data_prep.py first.")
    df = pd.read_parquet(split_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    all_preds, all_labels = [], []
    texts = df["text"].tolist()
    labels = df["label_id"].tolist()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts, truncation=True, padding=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels[i:i + batch_size])

    report = classification_report(
        all_labels, all_preds, target_names=LABELS, zero_division=0, digits=3
    )
    report_dict = classification_report(
        all_labels, all_preds, target_names=LABELS, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(LABELS))))

    print(f"\n=== Classification report ({split}) ===\n{report}")
    print("=== Confusion matrix (rows=true, cols=pred) ===")
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    print(cm_df.to_string())

    out_dir = Path(model_dir)
    (out_dir / f"eval_report_{split}.json").write_text(json.dumps(report_dict, indent=2))
    cm_df.to_csv(out_dir / f"confusion_matrix_{split}.csv")
    log.info("Saved eval artifacts to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=str(FINAL_MODEL_DIR))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    run_eval(Path(args.model_dir), args.split)
