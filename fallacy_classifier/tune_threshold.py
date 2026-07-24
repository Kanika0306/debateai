"""
Tunes the confidence threshold used by LocalFallacyAgent to decide
"trust the local DeBERTa prediction" vs "fall back to the LLM".

You're currently hardcoded at 0.65. This sweeps a range of thresholds
against the VAL set (never the test set — that stays untouched for the
final honest number) and reports, per threshold:

  - coverage: % of examples the local model would handle without fallback
  - local accuracy: accuracy ONLY on the examples above threshold
  - macro-F1 on the covered subset

The right threshold is a business tradeoff, not a pure metric:
  - Lower threshold -> more coverage (cheaper, faster) but more of the
    local model's mistakes reach the JudgeAgent unfiltered.
  - Higher threshold -> fewer local mistakes leak through, but more
    segments fall back to the (slower, costlier) LLM.

Also breaks the sweep down PER CLASS, because a single global threshold
is usually wrong — your weak classes from error_analysis.py likely need
a higher bar than your strong ones (e.g. "no fallacy" is probably safe
at a much lower threshold than "equivocation" if that's a weak class).

Usage:
    python tune_threshold.py --model-dir models/fallacy-classifier-v1
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import LABELS, SPLITS_DIR, FINAL_MODEL_DIR, MAX_LENGTH

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

THRESHOLDS = np.round(np.arange(0.30, 0.96, 0.05), 2)


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

    return np.array(preds), np.array(confidences)


def sweep_global(df: pd.DataFrame, preds: np.ndarray, confidences: np.ndarray):
    labels = df["label_id"].to_numpy()
    rows = []
    for t in THRESHOLDS:
        mask = confidences >= t
        coverage = mask.mean()
        if mask.sum() == 0:
            rows.append({"threshold": t, "coverage": 0.0, "local_accuracy": float("nan"), "local_macro_f1": float("nan")})
            continue
        acc = accuracy_score(labels[mask], preds[mask])
        f1 = f1_score(labels[mask], preds[mask], average="macro", zero_division=0)
        rows.append({"threshold": t, "coverage": round(coverage, 3), "local_accuracy": round(acc, 4), "local_macro_f1": round(f1, 4)})
    return pd.DataFrame(rows)


def sweep_per_class(df: pd.DataFrame, preds: np.ndarray, confidences: np.ndarray):
    labels = df["label_id"].to_numpy()
    results = {}
    for class_id, class_name in enumerate(LABELS):
        class_mask = labels == class_id
        if class_mask.sum() == 0:
            continue
        rows = []
        for t in THRESHOLDS:
            covered = class_mask & (confidences >= t)
            coverage = covered.sum() / class_mask.sum()
            if covered.sum() == 0:
                rows.append({"threshold": t, "coverage": 0.0, "local_accuracy": float("nan")})
                continue
            correct = (preds[covered] == class_id).sum()
            acc = correct / covered.sum()
            rows.append({"threshold": t, "coverage": round(coverage, 3), "local_accuracy": round(acc, 4)})
        results[class_name] = pd.DataFrame(rows)
    return results


def recommend_threshold(global_df: pd.DataFrame, min_local_accuracy: float = 0.85):
    """Suggest the lowest threshold that still hits a target local accuracy —
    maximizes coverage (cost savings) subject to a quality floor."""
    candidates = global_df[global_df["local_accuracy"] >= min_local_accuracy]
    if candidates.empty:
        return None
    return candidates.sort_values("threshold").iloc[0]


def main(model_dir: Path, min_local_accuracy: float):
    val_path = SPLITS_DIR / "val.parquet"
    if not val_path.exists():
        raise FileNotFoundError(f"Missing {val_path}. Run data_prep.py first.")
    df = pd.read_parquet(val_path)

    preds, confidences = get_predictions(model_dir, df)

    print("\n=== Global threshold sweep (val set) ===")
    global_df = sweep_global(df, preds, confidences)
    print(global_df.to_string(index=False))

    rec = recommend_threshold(global_df, min_local_accuracy)
    if rec is not None:
        print(
            f"\nRecommended global threshold: {rec['threshold']} "
            f"(coverage={rec['coverage']:.1%}, local_accuracy={rec['local_accuracy']:.1%}, "
            f"target was >= {min_local_accuracy:.0%})"
        )
    else:
        print(
            f"\nNo threshold in the sweep range hits {min_local_accuracy:.0%} local accuracy. "
            "The model isn't reliable enough yet even at high confidence for a pure-threshold "
            "cutoff — run error_analysis.py and address the weak classes before tightening this further."
        )

    print("\n=== Per-class threshold sweep (val set) ===")
    per_class = sweep_per_class(df, preds, confidences)
    for class_name, class_df in per_class.items():
        rec_c = recommend_threshold(class_df, min_local_accuracy)
        rec_str = f"-> recommend {rec_c['threshold']}" if rec_c is not None else "-> no threshold hits target, keep on LLM fallback"
        print(f"\n{class_name} {rec_str}")
        print(class_df.to_string(index=False))

    out_path = Path(model_dir) / "threshold_sweep_val.csv"
    global_df.to_csv(out_path, index=False)
    log.info("\nSaved global sweep table -> %s", out_path)
    print(
        "\nNote: per-class thresholds require changing LocalFallacyAgent from a single float\n"
        "`threshold` to a dict keyed by predicted label. Wire it in inference.py's\n"
        "analyze_sync() as: `if conf < self.threshold.get(label, self.default_threshold): continue`"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=str(FINAL_MODEL_DIR))
    parser.add_argument("--min-local-accuracy", type=float, default=0.85,
                        help="Quality floor for the local model before it's trusted without LLM fallback.")
    args = parser.parse_args()
    main(Path(args.model_dir), args.min_local_accuracy)
