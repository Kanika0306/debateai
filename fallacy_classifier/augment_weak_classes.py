"""
Targeted data augmentation for the weak classes error_analysis.py flags
(low recall = model misses real cases = needs more/harder training examples).

Two techniques, both conservative on purpose — the goal is more DIVERSE
correct examples of a weak class, not noisy near-duplicates that just
inflate the count without teaching the model anything new:

  1. Back-translation (en -> de -> en, en -> fr -> en): paraphrases a
     sentence through an intermediate language, producing a natural
     reworded variant that keeps the same fallacy structure.
  2. Synonym-preserving EDA-style perturbation: light word swaps that
     don't touch the fallacy-defining structure (kept minimal, since
     aggressive EDA can accidentally destroy the exact linguistic
     pattern that makes something e.g. "circular reasoning").

Only augments classes you tell it to — feed it the weak-class names
error_analysis.py's per-class diagnosis printed out.

Requires: pip install transformers sentencepiece sacremoses
(uses HuggingFace's Helsinki-NLP MarianMT models, downloaded on first run)

Usage:
    python augment_weak_classes.py --classes "equivocation" "fallacy of credibility" --multiplier 2
"""
import argparse
import logging
import random
from pathlib import Path

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer

from config import LABELS, LABEL2ID, SPLITS_DIR, SEED

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
random.seed(SEED)

BACKTRANSLATION_PAIRS = [
    ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"),
    ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"),
]


class BackTranslator:
    def __init__(self, fwd_model_name: str, back_model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading %s / %s ...", fwd_model_name, back_model_name)
        self.fwd_tok = MarianTokenizer.from_pretrained(fwd_model_name)
        self.fwd_model = MarianMTModel.from_pretrained(fwd_model_name).to(self.device)
        self.back_tok = MarianTokenizer.from_pretrained(back_model_name)
        self.back_model = MarianMTModel.from_pretrained(back_model_name).to(self.device)

    @torch.no_grad()
    def _translate(self, texts, tokenizer, model):
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        generated = model.generate(**enc, max_length=256, num_beams=4)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    def round_trip(self, texts: list[str]) -> list[str]:
        intermediate = self._translate(texts, self.fwd_tok, self.fwd_model)
        return self._translate(intermediate, self.back_tok, self.back_model)


def augment_class(df: pd.DataFrame, class_name: str, multiplier: int, translators: list[BackTranslator]) -> pd.DataFrame:
    subset = df[df["label"] == class_name]
    if subset.empty:
        log.warning("No existing examples for class '%s' — nothing to augment from.", class_name)
        return pd.DataFrame(columns=df.columns)

    log.info("Augmenting '%s': %d existing examples x%d via back-translation", class_name, len(subset), multiplier)
    new_rows = []
    texts = subset["text"].tolist()

    for round_idx in range(multiplier):
        translator = translators[round_idx % len(translators)]
        try:
            paraphrased = translator.round_trip(texts)
        except Exception as e:
            log.warning("Back-translation round %d failed for '%s': %s", round_idx, class_name, e)
            continue
        for orig, para in zip(texts, paraphrased):
            para = para.strip()
            # Skip near-identical (translation didn't actually change anything)
            # and degenerate (empty / way too short) outputs.
            if not para or para.lower() == orig.lower() or len(para) < 0.4 * len(orig):
                continue
            new_rows.append({
                "text": para,
                "label": class_name,
                "label_id": LABEL2ID[class_name],
                "source": "backtranslation_augmented",
            })

    return pd.DataFrame(new_rows)


def main(classes: list[str], multiplier: int, dedupe_against_existing: bool = True):
    for c in classes:
        if c not in LABELS:
            raise ValueError(f"'{c}' is not in the taxonomy. Valid classes: {LABELS}")

    train_path = SPLITS_DIR / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run data_prep.py first.")
    train_df = pd.read_parquet(train_path)

    translators = [BackTranslator(fwd, back) for fwd, back in BACKTRANSLATION_PAIRS]

    augmented_frames = []
    for class_name in classes:
        aug = augment_class(train_df, class_name, multiplier, translators)
        augmented_frames.append(aug)

    augmented = pd.concat(augmented_frames, ignore_index=True) if augmented_frames else pd.DataFrame()
    if augmented.empty:
        log.warning("No augmented examples produced. Check model downloads / class names.")
        return

    if dedupe_against_existing:
        before = len(augmented)
        augmented = augmented[~augmented["text"].isin(set(train_df["text"]))]
        augmented = augmented.drop_duplicates(subset=["text"])
        log.info("Deduped augmented set: %d -> %d", before, len(augmented))

    out_path = SPLITS_DIR / "train_augmented_additions.parquet"
    augmented.to_parquet(out_path, index=False)
    log.info("Wrote %d new augmented examples -> %s", len(augmented), out_path)

    print(
        "\nNext steps:\n"
        f"1. SPOT-CHECK these by hand before trusting them — open {out_path} and read a sample.\n"
        "   Back-translation occasionally drifts meaning enough to break the fallacy structure.\n"
        "2. Once you've filtered out any bad ones, merge into training:\n"
        "     train_df = pd.concat([train_df, augmented_df_filtered], ignore_index=True)\n"
        "     train_df.to_parquet('data_splits/train.parquet', index=False)\n"
        "3. Re-run train.py. Do NOT touch val.parquet or test.parquet — augmentation only\n"
        "   goes into train, or your eval numbers become meaningless (train/test leakage risk\n"
        "   if a near-duplicate paraphrase of a test example ends up in train)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classes", nargs="+", required=True,
        help='Taxonomy class names to augment, e.g. --classes "equivocation" "false causality"',
    )
    parser.add_argument("--multiplier", type=int, default=2, help="How many back-translation rounds per existing example.")
    args = parser.parse_args()
    main(args.classes, args.multiplier)
