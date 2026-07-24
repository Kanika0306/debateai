"""
Shared config for the Fallacy Classifier training pipeline.
Keep label order stable — it's baked into the saved model's config.json
and MUST match what inference.py expects.
"""
from pathlib import Path

# ---- Label taxonomy (must match PROJECT_DOCUMENTATION.md exactly) ----
LABELS = [
    "ad hominem",
    "ad populum",
    "appeal to emotion",
    "circular reasoning",
    "false causality",
    "false dilemma",
    "hasty generalization",
    "fallacy of relevance",
    "fallacy of credibility",
    "equivocation",
    "no fallacy",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# ---- Paths (match your repo's data/ layout) ----
REPO_ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = REPO_ROOT.parent / "data" / "processed"
DATA_RAW = REPO_ROOT.parent / "data" / "raw"

FALLACIES_PARQUET = DATA_PROCESSED / "fallacies_parsed.parquet"
FALLACIES_JSONL = DATA_PROCESSED / "fallacies" / "fallacy_examples_unified.jsonl"
ARGOTARIO_DIR = DATA_RAW / "fallacies"  # expects *.tsv files here

SPLITS_DIR = REPO_ROOT / "data_splits"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "fallacy-classifier"
FINAL_MODEL_DIR = REPO_ROOT / "models" / "fallacy-classifier-v1"

# ---- Model / training hyperparameters ----
BASE_MODEL = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
SEED = 42

TRAIN_ARGS = dict(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=6,
    weight_decay=0.01,
    warmup_ratio=0.06,
    gradient_accumulation_steps=2,
    fp16=False,  # DeBERTa-v3 requires bf16 or fp32 to prevent FP16 gradient overflow
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=25,
    save_total_limit=2,
    report_to=[],
)
