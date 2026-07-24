"""
Loads + normalizes fallacy data from repository sources into one canonical schema:
    text: str, label: str (one of config.LABELS)

Sources:
  1. data/processed/fallacies/fallacy_examples_unified.jsonl (or data/processed/fallacies_parsed.parquet)
  2. data/raw/fallacies/**/*.tsv and data/raw/fallacies/**/*.parquet (Argotario + logic_dataset)

Produces stratified train/val/test parquet splits under SPLITS_DIR.
Run standalone: python data_prep.py
"""
import logging
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    LABELS, LABEL2ID, FALLACIES_PARQUET, FALLACIES_JSONL, ARGOTARIO_DIR, SPLITS_DIR, SEED,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Maps source label strings -> canonical 11-class taxonomy.
LABEL_ALIASES = {
    "ad hominem": "ad hominem",
    "personal attack": "ad hominem",
    "ad populum": "ad populum",
    "appeal to popularity": "ad populum",
    "bandwagon": "ad populum",
    "appeal to emotion": "appeal to emotion",
    "emotional appeal": "appeal to emotion",
    "appeal to fear": "appeal to emotion",
    "circular reasoning": "circular reasoning",
    "circular argument": "circular reasoning",
    "begging the question": "circular reasoning",
    "fallacy of logic": "circular reasoning",
    "false causality": "false causality",
    "false cause": "false causality",
    "post hoc": "false causality",
    "false dilemma": "false dilemma",
    "false dichotomy": "false dilemma",
    "black-or-white": "false dilemma",
    "hasty generalization": "hasty generalization",
    "overgeneralization": "hasty generalization",
    "faulty generalization": "hasty generalization",
    "fallacy of relevance": "fallacy of relevance",
    "red herring": "fallacy of relevance",
    "irrelevant conclusion": "fallacy of relevance",
    "fallacy of extension": "fallacy of relevance",
    "intentional": "fallacy of relevance",
    "fallacy of credibility": "fallacy of credibility",
    "appeal to authority": "fallacy of credibility",
    "false authority": "fallacy of credibility",
    "irrelevant authority": "fallacy of credibility",
    "equivocation": "equivocation",
    "ambiguity": "equivocation",
    "no fallacy": "no fallacy",
    "none": "no fallacy",
    "valid": "no fallacy",
    "not a fallacy": "no fallacy",
}


def _normalize_label(raw: str) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    key = re.sub(r"[_\-]+", " ", raw.strip().lower())
    key = re.sub(r"\s+", " ", key)
    return LABEL_ALIASES.get(key)


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_processed_fallacies() -> pd.DataFrame:
    """Loads unified jsonl or parsed parquet from data/processed/."""
    target_path = None
    if FALLACIES_JSONL.exists():
        target_path = FALLACIES_JSONL
        df = pd.read_json(target_path, lines=True)
    elif FALLACIES_PARQUET.exists():
        target_path = FALLACIES_PARQUET
        df = pd.read_parquet(target_path)
    else:
        log.warning("Neither %s nor %s found — skipping processed fallacies.", FALLACIES_JSONL, FALLACIES_PARQUET)
        return pd.DataFrame(columns=["text", "label"])

    text_col = _pick_col(df, ["text", "sentence", "source_article", "content", "argument", "Text"])
    label_col = _pick_col(df, ["label", "fallacy_type", "class", "fallacy", "Intended Fallacy", "logical_fallacies"])

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not find text/label columns in {target_path}. "
            f"Found columns: {list(df.columns)}."
        )

    out = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    raw_labels = out["label"].dropna().unique()
    out["label"] = out["label"].map(_normalize_label)
    
    unmapped = [r for r in raw_labels if _normalize_label(r) is None]
    if unmapped:
        log.warning("Unmapped raw labels in %s: %s", target_path.name, unmapped)

    out = out.dropna(subset=["text", "label"])
    out["source"] = target_path.name
    return out


def load_raw_fallacies() -> pd.DataFrame:
    """Loads fallback TSV/Parquet files from data/raw/fallacies/ if available."""
    if not ARGOTARIO_DIR.exists():
        return pd.DataFrame(columns=["text", "label"])

    files = sorted(list(ARGOTARIO_DIR.rglob("*.tsv")) + list(ARGOTARIO_DIR.rglob("*.parquet")))
    if not files:
        return pd.DataFrame(columns=["text", "label"])

    frames = []
    for f in files:
        try:
            if f.suffix == ".tsv":
                df = pd.read_csv(f, sep="\t")
            else:
                df = pd.read_parquet(f)
        except Exception as e:
            log.warning("Failed to read %s: %s", f, e)
            continue

        text_col = _pick_col(df, ["text", "Text", "argument", "sentence", "source_article"])
        label_col = _pick_col(df, ["label", "Intended Fallacy", "fallacy", "Fallacy", "logical_fallacies", "fallacy_type"])
        if text_col is None or label_col is None:
            continue

        sub = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
        sub["label"] = sub["label"].map(_normalize_label)
        sub = sub.dropna(subset=["text", "label"])
        sub["source"] = f.name
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["text", "label"])
    return pd.concat(frames, ignore_index=True)


def build_dataset() -> pd.DataFrame:
    a = load_processed_fallacies()
    b = load_raw_fallacies()
    df = pd.concat([a, b], ignore_index=True)

    before = len(df)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text", "label"])
    log.info("Combined dataset: %d rows (from %d before cleaning)", len(df), before)

    counts = df["label"].value_counts()
    log.info("Final class distribution:\n%s", counts.to_string())

    missing_classes = set(LABELS) - set(df["label"].unique())
    if missing_classes:
        log.error("CRITICAL: No examples found for classes: %s", missing_classes)
    else:
        log.info("All 11 taxonomy classes are present in the dataset.")

    df["label_id"] = df["label"].map(LABEL2ID)
    return df.reset_index(drop=True)


def make_splits(df: pd.DataFrame, test_size=0.15, val_size=0.15):
    counts = df["label"].value_counts()
    rare = counts[counts < 3].index.tolist()
    if rare:
        log.warning("Classes with <3 examples: %s.", rare)

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=df["label_id"]
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        random_state=SEED,
        stratify=train_val["label_id"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main():
    df = build_dataset()
    if df.empty:
        log.error("No data loaded. Check data sources.")
        return

    train, val, test = make_splits(df)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(SPLITS_DIR / "train.parquet", index=False)
    val.to_parquet(SPLITS_DIR / "val.parquet", index=False)
    test.to_parquet(SPLITS_DIR / "test.parquet", index=False)
    log.info(
        "Wrote splits -> train=%d val=%d test=%d to %s",
        len(train), len(val), len(test), SPLITS_DIR,
    )


if __name__ == "__main__":
    main()
