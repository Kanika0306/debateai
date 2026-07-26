# Fact Verification (NLI) — Model 2

DeBERTa-v3-large fine-tuned on FEVER + LIAR to predict:
`SUPPORTS` / `REFUTES` / `NOT_ENOUGH_INFO` for a (claim, evidence) pair.

## Run order
```
pip install -r requirements.txt
python data_prep.py
python train.py
python evaluate.py
```

## Before you train
1. Drop your FEVER export (jsonl, with claim + label + evidence text) and
   LIAR file (tsv/csv) under `./raw_data/`.
2. Run `data_prep.py` and **read its printed warnings** — it will tell you
   about unmapped labels or zero-example classes. Do not proceed to
   `train.py` until those are resolved.
3. Note the label-collapsing assumption in `config.py`
   (`liar_label_map`): LIAR's 6-way truthfulness scale is merged into the
   3-class FEVER scheme. If you want LIAR's finer granularity preserved,
   that needs a separate multi-task head — flag it if that's a requirement.

## Key differences from the fallacy classifier
- This is a **pair-input** task (premise/evidence + hypothesis/claim), not
  single-sentence classification — tokenization passes both fields.
- LIAR has no retrieved evidence, so its rows train the model on
  hypothesis-only signal (empty premise). This is a known weak spot: a model
  trained partly on hypothesis-only examples can learn to shortcut on claim
  wording alone. Watch the `NOT_ENOUGH_INFO` F1 in `evaluate.py`'s report
  specifically — that's usually where this shortcut shows up as
  overconfident SUPPORTS/REFUTES calls when it should have deferred.
- Confidence gate is set at 0.70 (vs. 0.65 for the fallacy model) since a
  wrong "REFUTES" verdict surfaced to a user is a costlier mistake than a
  wrong fallacy tag.

## Wiring in
`inference.py` exposes `LocalFactVerificationAgent.verify(claim, evidence)`,
async, confidence-gated — returns `None` under threshold so your
orchestrator can fall back to the LLM verifier. **Match its signature to
your actual `FactVerificationAgent`/`JudgeAgent` contract before wiring it
in** — I don't have your real interface in this session, so I built to the
same pattern as the fallacy agent rather than guessing your exact method
names.
