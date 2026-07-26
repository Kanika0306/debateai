"""
Fine-tunes DeBERTa-v3-large as a (premise, hypothesis) -> 3-class NLI head
for fact verification. Uses inverse-frequency class weighting and tracks
macro-F1 for checkpoint selection, same convention as the fallacy classifier.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback,
    DataCollatorWithPadding
)
from config import CFG

LABEL2ID = {l: i for i, l in enumerate(CFG.labels)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def load_split(name):
    df = pd.read_csv(os.path.join(CFG.processed_dir, f"{name}.csv"))
    df["premise"] = df["premise"].fillna("")
    df["hypothesis"] = df["hypothesis"].fillna("")
    df["labels"] = df["label"].map(LABEL2ID)
    return Dataset.from_pandas(df[["premise", "hypothesis", "labels"]], preserve_index=False)


def tokenize_fn(tokenizer):
    def _fn(batch):
        return tokenizer(
            batch["premise"], batch["hypothesis"],
            truncation=True, max_length=CFG.max_length, padding="max_length"
        )
    return _fn


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    p, r, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(len(CFG.labels)))
    )
    metrics = {"macro_f1": macro_f1}
    for i, lbl in ID2LABEL.items():
        metrics[f"f1_{lbl}"] = f1_per_class[i]
    return metrics


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    torch.manual_seed(CFG.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model_name, num_labels=len(CFG.labels),
        id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_ds = load_split("train")
    val_ds = load_split("val")
    if len(val_ds) > 5000:
        val_ds = val_ds.select(range(5000))

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    print("Tokenizing datasets...", flush=True)
    tok_fn = tokenize_fn(tokenizer)
    train_ds = train_ds.map(tok_fn, batched=True)
    val_ds = val_ds.map(tok_fn, batched=True)
    print(f"Tokenization complete! Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    class_weights = None
    if CFG.use_class_weights:
        y = train_ds["labels"]
        weights = compute_class_weight(
            class_weight="balanced", classes=np.arange(len(CFG.labels)), y=y
        )
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print("Class weights:", dict(zip(CFG.labels, weights)), flush=True)

    args = TrainingArguments(
        output_dir=CFG.output_dir,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size * 2,
        gradient_accumulation_steps=CFG.grad_accum_steps,
        learning_rate=CFG.learning_rate,
        num_train_epochs=CFG.num_epochs,
        warmup_ratio=CFG.warmup_ratio,
        weight_decay=CFG.weight_decay,
        fp16=CFG.fp16 and torch.cuda.is_available(),
        bf16=getattr(CFG, "bf16", False) and torch.cuda.is_available(),
        # Chunked training: eval/save every `eval_steps`, not once per epoch.
        # This surfaces accuracy regressions mid-run and keeps the
        # best-by-macro-F1 checkpoint instead of just the last one.
        eval_strategy="steps",
        eval_steps=CFG.eval_steps,
        save_strategy="steps",
        save_steps=CFG.eval_steps,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=3,
        report_to=[],
    )

    class LogMetricsCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            step = state.global_step
            macro_f1 = metrics.get("eval_macro_f1", 0.0)
            f1_sup = metrics.get("eval_f1_SUPPORTS", 0.0)
            f1_ref = metrics.get("eval_f1_REFUTES", 0.0)
            f1_nei = metrics.get("eval_f1_NOT_ENOUGH_INFO", 0.0)
            print(f"\n[CHUNK EVAL @ Step {step}] Macro-F1: {macro_f1:.4f} | SUPPORTS: {f1_sup:.4f} | REFUTES: {f1_ref:.4f} | NOT_ENOUGH_INFO: {f1_nei:.4f}\n", flush=True)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[LogMetricsCallback(), EarlyStoppingCallback(early_stopping_patience=20)],
    )

    trainer.train()

    print("\n=== Macro-F1 by chunk (every eval_steps) ===")
    for entry in trainer.state.log_history:
        if "eval_macro_f1" in entry:
            print(f"  step {entry.get('step')}: macro_f1={entry['eval_macro_f1']:.4f}")

    trainer.save_model(os.path.join(CFG.output_dir, "best"))
    tokenizer.save_pretrained(os.path.join(CFG.output_dir, "best"))
    print(f"\nBest checkpoint (by macro_f1, not just last step) saved to {CFG.output_dir}/best")


if __name__ == "__main__":
    main()
