"""
Fine-tunes DeBERTa-v3-base as an 11-class fallacy classifier.

Usage:
    python data_prep.py        # once, to build splits
    python train.py            # trains + saves to config.FINAL_MODEL_DIR
"""
import logging

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

from config import (
    LABELS, LABEL2ID, ID2LABEL, NUM_LABELS,
    BASE_MODEL, MAX_LENGTH, SEED,
    SPLITS_DIR, CHECKPOINT_DIR, FINAL_MODEL_DIR, TRAIN_ARGS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

torch.manual_seed(SEED)


def load_splits():
    for name in ("train", "val", "test"):
        p = SPLITS_DIR / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run `python data_prep.py` first.")
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val = pd.read_parquet(SPLITS_DIR / "val.parquet")
    test = pd.read_parquet(SPLITS_DIR / "test.parquet")
    return train, val, test


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM_LABELS)), zero_division=0
    )
    metrics = {
        "macro_f1": macro_f1,
        "eval_macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
    }
    for i, label in enumerate(LABELS):
        safe = label.replace(" ", "_")
        metrics[f"f1_{safe}"] = f1_per_class[i]
    return metrics


def main():
    train_df, val_df, test_df = load_splits()
    log.info("Loaded splits: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding=False)

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    val_ds = Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    test_ds = Dataset.from_pandas(test_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args_dict = dict(TRAIN_ARGS)
    args_dict["fp16"] = False
    args_dict["bf16"] = False  # DeBERTa-v3 is most stable in FP32 on PyTorch
    args_dict["adam_epsilon"] = 1e-6
    args_dict["max_grad_norm"] = 1.0
    args_dict["learning_rate"] = 3e-5
    args_dict["num_train_epochs"] = 4

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        seed=SEED,
        **args_dict,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    log.info("Starting fine-tuning on %s", "GPU" if torch.cuda.is_available() else "CPU")
    trainer.train()

    log.info("Evaluating on held-out test set...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    log.info("Test metrics: %s", test_metrics)

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

    with open(FINAL_MODEL_DIR / "test_metrics.txt", "w") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")

    log.info("Saved final model to %s", FINAL_MODEL_DIR)


if __name__ == "__main__":
    main()
