# Fallacy Classifier — Training Pipeline

Fine-tunes `microsoft/deberta-v3-base` into an 11-class logical fallacy classifier,
trained on your `fallacies_parsed.parquet` + Argotario TSVs.
Runs fine on a single 6 GB+ VRAM local GPU.

---

## Setup

```bash
pip install -r requirements.txt
```

Place this folder as a sibling of your `data/` directory (matching `config.py`'s expected layout),
or edit the paths in `config.py` directly:

```text
your-repo/
├── data/
│   ├── processed/fallacies_parsed.parquet
│   └── raw/fallacies/*.tsv          # Argotario
└── fallacy_classifier/              # this folder
```

---

## Run Order

```bash
python data_prep.py     # 1. builds normalized, stratified train/val/test splits
python train.py         # 2. fine-tunes DeBERTa-v3-base, saves best checkpoint
python evaluate.py      # 3. full classification report + confusion matrix
```

`data_prep.py` logs the class distribution and warns you about:
- label strings it couldn't map to the 11-class taxonomy (extend `LABEL_ALIASES` in `data_prep.py` when you see these)
- classes with too few examples (`< 3`) to stratify safely
- taxonomy classes with zero training examples (the model literally cannot learn these — check your source data before training)

`train.py` uses `adam_epsilon=1e-6` and `max_grad_norm=1.0` to prevent the DeBERTa-v3 relative position attention instability that causes `NaN` loss under mixed precision.

Training tracks **macro-F1** (not accuracy) as the model-selection metric, because accuracy on an imbalanced 11-class set is misleading — macro-F1 forces every class to actually be learned.

---

## Using the Trained Model in Your Pipeline

```python
from inference import LocalFallacyAgent

agent = LocalFallacyAgent(threshold=0.55)  # tune threshold on your val set
flags = await agent.analyze(segment_text)
# -> [{"text": "...", "fallacy_type": "ad hominem", "confidence": 0.87}]
```

This matches your existing `FallacyAgent` contract (segment text in, flagged spans + taxonomy
classification out), so you can:
- **Replace** the LLM-prompted agent outright (cheaper, faster, no API dependency), or
- **Ensemble**: run both, only escalate to the LLM when the local model's confidence is below
  threshold, or when it and the LLM disagree — cuts API cost while keeping LLM-quality judgment
  for ambiguous cases.

---

## Threshold Tuning

The default `threshold=0.55` is a starting point, not a final answer. After training, sweep it
against `data_splits/val.parquet` and pick the value that gives you the precision/recall tradeoff
you need. `LocalFallacyAgent` now accepts a **per-class threshold dict** for fine-grained control:

```python
agent = LocalFallacyAgent(threshold={
    "no fallacy": 0.45,
    "equivocation": 0.75,   # weak class -> higher bar before trusting local model
    "_default": 0.65,
})
```

---

## Fine-Tuning Loop (Post-Deployment Improvement)

Now that v1 is trained, ensembled, and passing your integration suite, use this loop to improve it
rather than guessing at changes:

```bash
python error_analysis.py                                          # 1. find what's actually wrong
python tune_threshold.py                                          # 2. fix the confidence cutoff
python augment_weak_classes.py --classes "..." --multiplier 2    # 3. fix the data
python train.py                                                   # 4. retrain
python evaluate.py                                                # 5. confirm improvement on test
```

### 1. `error_analysis.py`
Pulls the real misclassified sentences (not just confusion-matrix counts) grouped by
`(true → predicted)` pair, ranked so you read the highest-confidence mistakes first — those are
the most informative, since the model was **sure and still wrong**. Also gives a per-class
recall/precision diagnosis so you know whether a weak class is weak because:
- **Low recall** → model misses real cases → needs more training data
- **Low precision** → model over-fires → check for label ambiguity in source datasets

### 2. `tune_threshold.py`
Sweeps confidence thresholds on the val set (**never test**) and reports coverage vs. local
accuracy at each one, both globally and per class. Your hardcoded `0.65` was a starting guess;
this tells you the actual tradeoff curve. Usually recommends a **higher threshold for weak classes**
and a **lower one for strong classes** like `no fallacy`.

### 3. `augment_weak_classes.py`
For classes `error_analysis.py` flags as low-recall, generates **back-translated paraphrases**
(`en→de→en`, `en→fr→en`) of existing examples to add genuine linguistic diversity — not just
duplicate counts. Writes to a separate file (`data_splits/train_augmented_additions.parquet`);
**spot-check the output by hand** before merging into `train.parquet`, and **never touch
`val`/`test` with augmented data** or your eval numbers stop meaning anything.

### 4–5. Retrain and Re-evaluate
Compare the new `eval_report_test.json` against your saved one from v1 to confirm the changes
actually helped before deploying.

---

## Next Model in the Roadmap

Once this is trained and validated in your pipeline, the next highest-ROI target is the
**Fact Verification NLI model** (`FEVER + LIAR → deberta-v3-large`, same class-weighting +
macro-F1 approach applies).
