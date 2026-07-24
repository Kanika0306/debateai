"""
Error analysis for the trained fallacy classifier.

Goes beyond the aggregate confusion_matrix_test.csv / eval_report_test.json
you already have — pulls out the ACTUAL misclassified sentences, grouped by
(true_label -> predicted_label) pair, ranked by frequency, so you can read
real examples instead of just staring at numbers.

This is the step that tells you WHY a class is weak: is it genuinely
ambiguous data (two fallacies co-occurring in one sentence), a labeling
error in the source dataset, or a real model blind spot that needs more
training examples.

Usage:
    python error_analysis.py --model-dir models/fallacy-classifier-v1 --split test
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import LABELS, SPLITS_DIR, FINAL_MODEL_DIR, MAX_LENGTH

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def get_predictions(model_dir: Path, df: pd.DataFrame, batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    texts = df["text"].tolist()
    preds, confidences = [], []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred_id = torch.max(probs, dim=-1)
            preds.extend(pred_id.cpu().tolist())
            confidences.extend(conf.cpu().tolist())

    return preds, confidences


def analyze(model_dir: Path, split: str, top_n_pairs: int = 10, examples_per_pair: int = 8):
    split_path = SPLITS_DIR / f"{split}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}. Run data_prep.py first.")
    df = pd.read_parquet(split_path)

    preds, confidences = get_predictions(model_dir, df)
    df = df.copy()
    df["pred_id"] = preds
    df["pred_label"] = [LABELS[p] for p in preds]
    df["confidence"] = confidences

    errors = df[df["label"] != df["pred_label"]].copy()
    log.info("Total errors: %d / %d (%.1f%% error rate)", len(errors), len(df), 100 * len(errors) / len(df))

    # Rank (true, pred) confusion pairs by frequency
    pair_counts = (
        errors.groupby(["label", "pred_label"]).size().reset_index(name="count").sort_values("count", ascending=False)
    )

    print("\n=== Top confusion pairs (true -> predicted) ===")
    print(pair_counts.head(top_n_pairs).to_string(index=False))

    # Dump real examples for the worst pairs, sorted by confidence
    # (high-confidence errors are the most informative — the model was SURE and wrong)
    out_rows = []
    for _, row in pair_counts.head(top_n_pairs).iterrows():
        true_l, pred_l, cnt = row["label"], row["pred_label"], row["count"]
        subset = errors[(errors["label"] == true_l) & (errors["pred_label"] == pred_l)]
        subset = subset.sort_values("confidence", ascending=False).head(examples_per_pair)
        for _, ex in subset.iterrows():
            out_rows.append({
                "true_label": true_l,
                "predicted_label": pred_l,
                "confidence": round(ex["confidence"], 4),
                "text": ex["text"],
                "source": ex.get("source", ""),
            })

    review_df = pd.DataFrame(out_rows)
    out_path = Path(model_dir) / f"error_analysis_{split}.csv"
    review_df.to_csv(out_path, index=False)
    log.info("Wrote %d high-confidence misclassified examples for manual review -> %s", len(review_df), out_path)

    # Per-class: is this class weak because of low recall (model misses it)
    # or low precision (model over-predicts it)?
    print("\n=== Per-class diagnosis ===")
    for label in LABELS:
        true_mask = df["label"] == label
        pred_mask = df["pred_label"] == label
        n_true = true_mask.sum()
        if n_true == 0:
            print(f"{label:26s} | NO TEST EXAMPLES — cannot evaluate")
            continue
        recall = (true_mask & pred_mask).sum() / n_true
        n_pred = pred_mask.sum()
        precision = (true_mask & pred_mask).sum() / n_pred if n_pred else float("nan")
        verdict = ""
        if recall < 0.6:
            verdict += " LOW RECALL (model misses real cases — needs more/harder training examples)"
        if n_pred and precision < 0.6:
            verdict += " LOW PRECISION (model over-fires — check for near-duplicate/ambiguous labels)"
        print(f"{label:26s} | recall={recall:.2f} precision={precision:.2f}{verdict}")

    print(
        f"\nFull review file: {out_path}\n"
        "Read these examples by hand. For each confusion pair, ask:\n"
        "  1. Is the TRUE label actually correct, or is this a source-dataset labeling error?\n"
        "  2. Do these two fallacy types genuinely overlap in this sentence (multi-label case\n"
        "     that a single-label classifier can't win)?\n"
        "  3. Is it a real gap — then go find/label more examples like this for the weak class."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=str(FINAL_MODEL_DIR))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--top-n-pairs", type=int, default=10)
    parser.add_argument("--examples-per-pair", type=int, default=8)
    args = parser.parse_args()
    analyze(Path(args.model_dir), args.split, args.top_n_pairs, args.examples_per_pair)
