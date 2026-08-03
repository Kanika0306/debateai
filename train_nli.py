"""
train_nli.py
=============
Fine-tunes roberta-large-mnli for 3-class fact verification:
    SUPPORTS / REFUTES / NEI (Not Enough Info)

Designed for 8-10 GB VRAM — uses gradient checkpointing + fp16
to fit roberta-large in tight VRAM budgets.

USAGE:
    python train_nli.py \
        --data data/processed/fact_verification/nli_combined.parquet \
        --output ./checkpoints/nli_model \
        --epochs 3 \
        --batch-size 8 \
        --grad-accum 4 \
        --fp16

GPU MEMORY GUIDE:
    8-10 GB VRAM  -> --batch-size 8  --grad-accum 4   (effective batch=32)
    12-16 GB VRAM -> --batch-size 16 --grad-accum 2
    If OOM: reduce --batch-size to 4, increase --grad-accum to 8
"""

import argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)


LABEL_LIST = ["NEI", "REFUTES", "SUPPORTS"]   # sorted → deterministic ids
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}


def load_data(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_json(path, lines=True)

    assert "claim" in df.columns, "Expected a 'claim' column"
    assert "label" in df.columns, "Expected a 'label' column"

    df = df[["claim", "label"]].dropna()
    df = df[df["label"].isin(LABEL_LIST)].reset_index(drop=True)
    df["label_id"] = df["label"].map(LABEL2ID)
    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class = f1_score(labels, preds, average=None,
                         labels=[0, 1, 2], zero_division=0)
    return {
        "f1_macro":    macro_f1,
        "f1_NEI":      per_class[0],
        "f1_REFUTES":  per_class[1],
        "f1_SUPPORTS": per_class[2],
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
            if self.class_weights is not None else None,
            label_smoothing=0.05,
        )
        loss = loss_fct(
            logits.view(-1, model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/processed/fact_verification/nli_combined.parquet")
    parser.add_argument("--output",     default="./checkpoints/nli_model")
    parser.add_argument("--base-model", default="roberta-large-mnli")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--grad-accum", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--max-length", type=int,   default=128)
    parser.add_argument("--fp16",       action="store_true")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--resume",     action="store_true", help="Resume training from latest checkpoint")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[warn] CUDA not found — will be very slow on CPU")
    else:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[info] GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {vram:.1f} GB")

    # ── Data ──────────────────────────────────────────────────
    print(f"\n[1/5] Loading {args.data}")
    df = load_data(args.data)
    print(f"  {len(df)} rows  |  {df['label'].value_counts().to_dict()}")

    train_df, val_df = train_test_split(
        df, test_size=0.10, random_state=args.seed, stratify=df["label_id"]
    )
    # Hold out 5% as a final test set
    val_df, test_df = train_test_split(
        val_df, test_size=0.33, random_state=args.seed, stratify=val_df["label_id"]
    )
    print(f"  Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    # ── Tokeniser ─────────────────────────────────────────────
    print(f"\n[2/5] Tokenising with {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    def make_ds(frame):
        ds = Dataset.from_pandas(
            frame[["claim", "label_id"]]
            .rename(columns={"claim": "text", "label_id": "labels"})
        )
        return ds.map(tokenize, batched=True, remove_columns=["text"])

    train_ds = make_ds(train_df)
    val_ds   = make_ds(val_df)
    test_ds  = make_ds(test_df)

    # ── Model ─────────────────────────────────────────────────
    print(f"\n[3/5] Loading {args.base_model}  (3-class head)")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,   # replaces MNLI 3-head with our 3-head
    )

    # Gradient checkpointing — critical for fitting in 8.5 GB
    model.gradient_checkpointing_enable()
    print("  Gradient checkpointing: ON")

    weights = compute_class_weight(
        "balanced",
        classes=np.arange(3),
        y=train_df["label_id"].values,
    )
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"  Class weights: {dict(zip(LABEL_LIST, weights.round(3)))}")

    # ── Training args ─────────────────────────────────────────
    print(f"\n[4/5] Training — {args.epochs} epochs | "
          f"batch {args.batch_size}×{args.grad_accum} | lr {args.lr}")

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        fp16=args.fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=100,
        seed=args.seed,
        report_to="none",
        # Keeps VRAM usage flat — important for 8.5 GB
        dataloader_pin_memory=False,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    if args.resume:
        print("\n[info] Resuming training from latest checkpoint in output_dir...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # ── Final test evaluation ──────────────────────────────────
    print(f"\n[5/5] Final test evaluation ({len(test_df)} held-out examples)")
    pred_out = trainer.predict(test_ds)
    preds    = np.argmax(pred_out.predictions, axis=-1)
    labels   = pred_out.label_ids

    report = classification_report(
        labels, preds,
        target_names=LABEL_LIST,
        digits=4,
    )
    print("\n" + "="*50)
    print("TEST SET REPORT")
    print("="*50)
    print(report)

    # Save
    final_path = f"{args.output}/best"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    import json, os
    os.makedirs(final_path, exist_ok=True)
    macro = f1_score(labels, preds, average="macro", zero_division=0)
    with open(f"{final_path}/eval_report.json", "w") as f:
        json.dump({
            "macro_f1": round(macro, 4),
            "report":   report,
            "label_map": LABEL2ID,
        }, f, indent=2)

    print(f"\nSaved -> {final_path}")
    print(f"Macro F1: {macro:.4f}")
    print("\nLoad later with:")
    print(f'  AutoModelForSequenceClassification.from_pretrained("{final_path}")')


if __name__ == "__main__":
    main()
