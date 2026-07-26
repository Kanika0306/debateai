# Task: Train and integrate the Fact Verification (NLI) model — Model 2

## Context
Model 1 (fallacy classifier) is done and live behind `FallacyAgent`. This
task trains Model 2: a DeBERTa-v3-large NLI model that verifies claims
against evidence (`SUPPORTS` / `REFUTES` / `NOT_ENOUGH_INFO`), trained on
FEVER + LIAR. The pipeline files are already dropped into
`fact_verification/` at the repo root.

## Scope — stay inside this only
Work only inside `fact_verification/` and the minimal integration point into
the real `FactVerificationAgent` / `JudgeAgent` contract. Do NOT touch the
other model directories, the fallacy classifier, or refactor unrelated code.

## Steps — checkpoint after each one, don't run straight through

### 1. Verify assumptions against the real repo
- Find the actual `FactVerificationAgent` (or equivalent) contract: method
  name, async or sync, exact input/output shape. `inference.py`'s
  `LocalFactVerificationAgent.verify(claim, evidence)` is a guess — confirm
  or correct it against real code, not against this prompt.
- Confirm your venv/Python version and GPU availability match what
  `requirements.txt` assumes.
- **Stop and report back before continuing if the contract doesn't match.**

### 2. File intake — copy, verify, then clean up
The real FEVER/LIAR files will be provided from wherever they currently sit
(uploads folder, another data dir, etc). Handle them exactly like this,
don't just reference them in place:
- Copy each source file into `fact_verification/raw_data/` as an **exact
  byte-for-byte copy** (e.g. `cp` / `shutil.copy2`, not a re-save or
  re-export that could silently change encoding, line endings, or drop
  rows).
- Verify the copy before trusting it: compare file size and a checksum
  (`sha256sum` or equivalent) between source and destination. If they don't
  match, stop and report — do not proceed on an unverified copy.
- Once verified, **delete the original source file** so there's a single
  source of truth under `raw_data/` and no stale duplicate elsewhere.
- Report back explicitly: which files were copied, their checksums matched,
  and that the originals were deleted. Don't delete anything without a
  confirmed-matching copy in place first.

### 3. Data validation
- Run `python data_prep.py`.
- Read the printed warnings closely: unmapped labels, zero-example classes,
  the label distribution. **Do not proceed to training if any class has
  near-zero examples or if a large fraction of rows were skipped** — report
  back with the numbers instead.

### 4. Train — in small chunks, not one long run
Don't run `train.py` as a single unmonitored pass. Split the run into small
chunks so accuracy problems surface early instead of at the end:
- Train in short chunks (e.g. a fraction of an epoch or a fixed step count —
  `config.py`'s `eval_strategy="epoch"` should be tightened to `"steps"`
  with a small `eval_steps` so you get a macro-F1 reading every chunk, not
  just once per epoch).
- After each chunk, check macro-F1 and per-class F1 before continuing to the
  next chunk. If a class (especially `NOT_ENOUGH_INFO`, the expected weak
  point given LIAR's hypothesis-only rows) is stalled or dropping, stop and
  report rather than burning the rest of the run on a bad trajectory.
- Keep the best-checkpoint-by-macro-F1 logic (`load_best_model_at_end`)
  active throughout so the smaller eval cadence actually pays off — the
  final saved model should be the best chunk, not just the last one.
- This chunked approach costs more wall-clock time than one long run but
  catches accuracy regressions (bad class weighting, a bad LR, a data issue
  that only shows up after N steps) while there's still budget to fix them.

### 5. Evaluate
- `python evaluate.py`
- Report macro-F1, per-class F1, confusion matrix, and the % of test
  predictions falling below the 0.70 confidence gate.
- **Checkpoint: do not integrate into the live pipeline until macro-F1 and
  per-class F1 are at a level you'd trust in production** — use the same bar
  you held Model 1 to. If accuracy is short of that bar, go back to chunked
  training rather than accepting a weak model to finish faster.

### 6. Integration
- Adjust `inference.py`'s method signature to exactly match the real
  `FactVerificationAgent` contract found in step 1.
- Wire `LocalFactVerificationAgent` in as a confidence-gated ensemble
  member alongside the existing LLM verifier, same pattern as
  `LocalFallacyAgent`.
- Update `README.md` / `PROJECT_DOCUMENTATION.md` to reflect the new model
  count and architecture, same as was done for Model 1.

## Definition of done
- Data validated with no unresolved warnings.
- Model trained, macro-F1 reported and judged acceptable by you (not just
  "it ran").
- `LocalFactVerificationAgent` wired into the real orchestrator with a
  confirmed-correct method signature.
- Docs updated, changes committed with a clear message.

Do not silently "fix" or retrain Model 1 while working on this. Do not move
on to Model 3 without being asked.
