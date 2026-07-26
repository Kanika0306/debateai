"""
Evaluates the best checkpoint on the held-out test split.
Reports macro-F1, per-class P/R/F1, and a confusion matrix — the same bar
you should hold every model in this project to before calling it done.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import CFG

LABELS = CFG.labels
CKPT = os.path.join(CFG.output_dir, "best")


def main():
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    df = pd.read_csv(os.path.join(CFG.processed_dir, "test.csv"))
    if len(df) > 3000:
        df = df.sample(n=3000, random_state=CFG.seed)
    premises = df["premise"].fillna("").tolist()
    hypotheses = df["hypothesis"].fillna("").tolist()

    all_preds, all_confs = [], []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            if i % (batch_size * 20) == 0:
                print(f"Evaluating test samples {i}/{len(df)}...", flush=True)
            p_batch = premises[i:i + batch_size]
            h_batch = hypotheses[i:i + batch_size]
            enc = tokenizer(
                p_batch, h_batch,
                truncation=True, max_length=CFG.max_length,
                padding=True, return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())

    y_true = df["label"].tolist()
    id2label = {i: l for i, l in enumerate(LABELS)}
    y_pred = [id2label[p] for p in all_preds]

    print("=== Classification report ===")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3))

    print("=== Confusion matrix (rows=true, cols=pred) ===")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))

    below_thresh = np.mean(np.array(all_confs) < CFG.confidence_threshold)
    print(f"\n{below_thresh*100:.1f}% of test predictions fall below the "
          f"{CFG.confidence_threshold} confidence gate and would defer to the LLM "
          f"fallback in production — sanity check this ratio isn't absurdly high.")

    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True)
    with open(os.path.join(CFG.output_dir, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {CFG.output_dir}/eval_report.json")


if __name__ == "__main__":
    main()
