# Fallacy Classifier — Training Pipeline

Fine-tunes `microsoft/deberta-v3-base` into an 11-class logical fallacy
classifier, trained on your `fallacies_parsed.parquet` + Argotario TSVs.
Runs fine on a single 6GB+ VRAM local GPU.

## Setup

```bash
pip install -r requirements.txt
```

Place this folder as a sibling of your `data/` directory (matching
`config.py`'s expected layout), or edit the paths in `config.py` directly:

```
your-repo/
├── data/
│   ├── processed/fallacies_parsed.parquet
│   └── raw/fallacies/*.tsv          # Argotario
└── fallacy_classifier/              # this folder
```

## Run order

```bash
python data_prep.py     # 1. builds normalized, stratified train/val/test splits
python train.py         # 2. fine-tunes DeBERTa-v3-base, saves best checkpoint
python evaluate.py      # 3. full classification report + confusion matrix
```

`data_prep.py` logs the class distribution and warns you about:
- label strings it couldn't map to the 11-class taxonomy (extend
  `LABEL_ALIASES` in `data_prep.py` when you see these)
- classes with too few examples (<3) to stratify safely
- taxonomy classes with **zero** training examples (the model literally
  cannot learn these — check your source data before training)

`train.py` uses inverse-frequency class weighting in the loss, since
`no fallacy` will dominate raw counts — otherwise the model just learns
to always predict "no fallacy" and looks falsely accurate.

Training tracks **macro-F1** (not accuracy) as the model-selection
metric, because accuracy on an imbalanced 11-class set is misleading —
macro-F1 forces every class to actually be learned, not just the
majority one.

## Using the trained model in your pipeline

```python
from inference import LocalFallacyAgent

agent = LocalFallacyAgent(threshold=0.55)  # tune threshold on your val set
flags = await agent.analyze(segment_text)
# -> [{"text": "...", "fallacy_type": "ad hominem", "confidence": 0.87}]
```

This matches your existing `FallacyAgent` contract (segment text in,
flagged spans + taxonomy classification out), so you can:
1. **Replace** the LLM-prompted agent outright (cheaper, faster, no API
   dependency), or
2. **Ensemble**: run both, only escalate to the LLM when the local
   model's confidence is below threshold, or when it and the LLM
   disagree — cuts API cost while keeping LLM-quality judgment calls
   for ambiguous cases.

## Threshold tuning

The default `threshold=0.55` is a starting point, not a final answer.
After training, sweep it against `data_splits/val.parquet` and pick the
value that gives you the precision/recall tradeoff you want (e.g. higher
threshold = fewer false-positive fallacy alerts in the live dashboard,
at the cost of missing some borderline cases).

## Next model in the roadmap

Once this is trained and validated in your pipeline, the next highest-ROI
target is the **Fact Verification NLI model** (FEVER + LIAR →
`deberta-v3-large`, same class-weighting + macro-F1 approach applies).
Say the word when you're ready and I'll build that pipeline the same way.
