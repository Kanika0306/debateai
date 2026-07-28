"""
train_claim_detector.py
==========================
Sequence Classification Fine-Tuning Script for DebateAI.
Fine-tunes transformer classifiers (Claim Detector, Fallacy Classifier, etc.).

Supports Parquet, CSV, and JSONL data files, class weighting, label smoothing (0.1),
custom metrics logging (macro F1, per-class F1), FP16 / BF16, and early stopping.

USAGE:
  python train_claim_detector.py \
    --data data/processed/fallacies/fallacy_examples_unified.jsonl \
    --text-col text \
    --label-col fallacy_type \
    --output ./checkpoints/fallacy_classifier \
    --base-model roberta-base \
    --epochs 8 \
    --batch-size 16 \
    --grad-accum 2 \
    --fp16
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class WeightedTrainer(Trainer):
    """Custom Trainer implementing weighted CrossEntropyLoss with Label Smoothing."""
    def __init__(self, class_weights=None, label_smoothing=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weight = self.class_weights.to(device=logits.device, dtype=logits.dtype) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.label_smoothing
        )

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred, id2label):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }

    for idx, label_name in id2label.items():
        metrics[f"f1_{label_name}"] = float(f1[idx])
        metrics[f"precision_{label_name}"] = float(precision[idx])
        metrics[f"recall_{label_name}"] = float(recall[idx])

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train DebateAI Sequence Classifier")
    parser.add_argument("--data", required=True, help="Path to input parquet, csv, or jsonl file")
    parser.add_argument("--text-col", default="text", help="Column name containing input text")
    parser.add_argument("--label-col", default="label", help="Column name containing labels")
    parser.add_argument("--output", default="./checkpoints/classifier", help="Output directory for model")
    parser.add_argument("--base-model", "--model-name", dest="model_name", default="roberta-base", help="Pretrained model identifier")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision training")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    set_seed(args.seed)

    print("============================================================")
    print("   DebateAI Sequence Classifier Training Pipeline")
    print("============================================================")
    print(f"Data file: {args.data}")
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Grad Accum: {args.grad_accum}")
    print(f"FP16: {args.fp16} | BF16: {args.bf16}")
    print("============================================================\n")

    data_path = args.data
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        sys.exit(1)

    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith(".jsonl") or data_path.endswith(".json"):
        df = pd.read_json(data_path, lines=True if data_path.endswith(".jsonl") else False)
    else:
        print("[ERROR] Unsupported data format. Use .parquet, .csv, or .jsonl")
        sys.exit(1)

    if args.text_col not in df.columns or args.label_col not in df.columns:
        print(f"[ERROR] Columns {args.text_col} and/or {args.label_col} not in DataFrame.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Clean data
    df = df.dropna(subset=[args.text_col, args.label_col]).reset_index(drop=True)
    df[args.text_col] = df[args.text_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(str)

    unique_labels = sorted(df[args.label_col].unique())
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for i, l in enumerate(unique_labels)}
    num_labels = len(unique_labels)

    print(f"Loaded {len(df)} records. Found {num_labels} classes: {label2id}")
    df["label_id"] = df[args.label_col].map(label2id)

    # Train / Val / Test Split (80% / 10% / 10%)
    train_df, test_val_df = train_test_split(
        df, test_size=0.20, random_state=args.seed, stratify=df["label_id"]
    )
    val_df, test_df = train_test_split(
        test_val_df, test_size=0.50, random_state=args.seed, stratify=test_val_df["label_id"]
    )

    print(f"Splits -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label_id"]),
        y=train_df["label_id"].values,
    )
    print(f"Class weights (balanced): {dict(zip(unique_labels, class_weights))}")

    # Tokenizer & Model
    print(f"\nLoading tokenizer & model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    train_encodings = tokenizer(
        train_df[args.text_col].tolist(), truncation=True, padding=True, max_length=args.max_length
    )
    val_encodings = tokenizer(
        val_df[args.text_col].tolist(), truncation=True, padding=True, max_length=args.max_length
    )
    test_encodings = tokenizer(
        test_df[args.text_col].tolist(), truncation=True, padding=True, max_length=args.max_length
    )

    train_dataset = TextClassificationDataset(train_encodings, train_df["label_id"].tolist())
    val_dataset = TextClassificationDataset(val_encodings, val_df["label_id"].tolist())
    test_dataset = TextClassificationDataset(test_encodings, test_df["label_id"].tolist())

    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    use_bf16 = args.bf16 and use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = args.fp16 and use_cuda and not use_bf16

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        label_smoothing=0.1,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=lambda ep: compute_metrics(ep, id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting model training...")
    trainer.train()

    print("\nEvaluating model on test set...")
    test_results = trainer.predict(test_dataset)
    test_preds = np.argmax(test_results.predictions, axis=1)

    report_str = classification_report(
        test_df["label_id"].values,
        test_preds,
        target_names=unique_labels,
        digits=4,
        zero_division=0,
    )
    print("\n================ TEST SET REPORT ================")
    print(report_str)

    # Save final best model & metadata
    best_model_path = os.path.join(out_dir, "best")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    metadata = {
        "label2id": label2id,
        "id2label": id2label,
        "test_metrics": test_results.metrics,
        "classification_report": report_str,
    }
    with open(os.path.join(best_model_path, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel training complete! Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
