"""
Loads FEVER (claim + evidence) and LIAR (claim + speaker statement) into a
single normalized (premise, hypothesis, label) NLI-style dataset.

Run this FIRST and read its printed report before training. It will warn you
about unmapped labels, empty splits, or missing evidence — do not train past
those warnings without fixing them.
"""
import os
import json
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from config import CFG


def _find_file(patterns, root):
    for pat in patterns:
        hits = glob.glob(os.path.join(root, "**", pat), recursive=True)
        if hits:
            return hits[0]
    return None


def extract_evidence_text(evidence):
    if evidence is None:
        return ""
    if isinstance(evidence, str):
        return evidence
    texts = []
    try:
        for item in evidence:
            if hasattr(item, "__getitem__") and len(item) >= 3:
                texts.append(str(item[2]))
            elif isinstance(item, str):
                texts.append(item)
    except Exception:
        pass
    return " ".join(texts)


def load_fever(root):
    search_dirs = [os.path.join(root, "fever"), root]
    path = None
    for sdir in search_dirs:
        path = _find_file(["*train*.parquet", "*.parquet", "*train*.jsonl", "*.jsonl"], sdir)
        if path:
            break
    if not path:
        print("[WARN] No FEVER dataset found under", root, "— skipping FEVER.")
        return pd.DataFrame(columns=["premise", "hypothesis", "label", "source"])

    rows = []
    skipped = 0

    if path.endswith(".parquet"):
        df_raw = pd.read_parquet(path)
        for _, row in df_raw.iterrows():
            claim = row.get("claim")
            label_raw = str(row.get("label", "")).strip().upper()
            evidence = extract_evidence_text(row.get("evidence"))
            label = CFG.fever_label_map.get(label_raw)
            if not claim or not label:
                skipped += 1
                continue
            rows.append({
                "premise": evidence or "",
                "hypothesis": claim,
                "label": label,
                "source": "fever",
            })
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                claim = obj.get("claim")
                label_raw = str(obj.get("label", "")).strip().upper()
                evidence = obj.get("evidence_text") or obj.get("evidence")
                if isinstance(evidence, list):
                    evidence = " ".join(str(e) for e in evidence if isinstance(e, str))
                label = CFG.fever_label_map.get(label_raw)
                if not claim or not label:
                    skipped += 1
                    continue
                rows.append({
                    "premise": evidence or "",
                    "hypothesis": claim,
                    "label": label,
                    "source": "fever",
                })
    print(f"[FEVER] loaded {len(rows)} rows from {path}, skipped {skipped} (bad/missing label or claim)")
    return pd.DataFrame(rows)


def load_liar(root):
    search_dirs = [os.path.join(root, "liar"), root]
    path = None
    for sdir in search_dirs:
        path = _find_file(["*train*.parquet", "*.parquet", "*train*.tsv", "*.tsv", "*train*.csv", "*.csv"], sdir)
        if path:
            break
    if not path:
        print("[WARN] No LIAR file found under", root, "— skipping LIAR.")
        return pd.DataFrame(columns=["premise", "hypothesis", "label", "source"])

    rows, skipped = [], 0

    if path.endswith(".parquet"):
        df_raw = pd.read_parquet(path)
        stmt_col = "statement" if "statement" in df_raw.columns else "claim"
        label_col = "label"
        for _, r in df_raw.iterrows():
            raw_label = r[label_col]
            stmt = r[stmt_col]
            label = CFG.liar_label_map.get(raw_label)
            if not label:
                try:
                    label = CFG.liar_label_map.get(int(raw_label))
                except (ValueError, TypeError):
                    pass
            if not label or not isinstance(stmt, str) or not stmt.strip():
                skipped += 1
                continue
            rows.append({
                "premise": "",  # LIAR has no retrieved evidence; hypothesis-only signal
                "hypothesis": stmt.strip(),
                "label": label,
                "source": "liar",
            })
    else:
        sep = "\t" if path.endswith(".tsv") else ","
        df = pd.read_csv(path, sep=sep, header=None, dtype=str, on_bad_lines="skip")

        if df.iloc[0].astype(str).str.contains("label|statement", case=False, na=False).any():
            df = pd.read_csv(path, sep=sep, dtype=str, on_bad_lines="skip")
            label_col = next((c for c in df.columns if "label" in c.lower()), None)
            stmt_col = next((c for c in df.columns if "statement" in c.lower() or "claim" in c.lower()), None)
        else:
            df.columns = [f"c{i}" for i in range(df.shape[1])]
            label_col, stmt_col = "c1", "c2"

        if label_col is None or stmt_col is None:
            print("[WARN] Could not detect LIAR label/statement columns — skipping LIAR.")
            return pd.DataFrame(columns=["premise", "hypothesis", "label", "source"])

        for _, r in df.iterrows():
            raw_label = str(r[label_col]).strip().lower()
            stmt = r[stmt_col]
            label = CFG.liar_label_map.get(raw_label)
            if not label or not isinstance(stmt, str) or not stmt.strip():
                skipped += 1
                continue
            rows.append({
                "premise": "",
                "hypothesis": stmt.strip(),
                "label": label,
                "source": "liar",
            })
    print(f"[LIAR] loaded {len(rows)} rows from {path}, skipped {skipped} (unmapped label / empty statement)")
    return pd.DataFrame(rows)


def main():
    os.makedirs(CFG.raw_data_dir, exist_ok=True)
    fever_df = load_fever(CFG.raw_data_dir)
    liar_df = load_liar(CFG.raw_data_dir)

    df = pd.concat([fever_df, liar_df], ignore_index=True)
    if df.empty:
        raise RuntimeError(
            f"No data loaded. Put FEVER jsonl and/or LIAR tsv under {CFG.raw_data_dir}/ and rerun."
        )

    print("\n=== Label distribution before split ===")
    print(df["label"].value_counts())
    zero_classes = [l for l in CFG.labels if l not in df["label"].unique()]
    if zero_classes:
        print(f"[WARN] These classes have ZERO examples: {zero_classes}. "
              f"Fix your data before training or the model will never predict them.")

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=CFG.seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=CFG.seed
    )

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(CFG.processed_dir, f"{name}.csv")
        split.to_csv(out_path, index=False)
        print(f"[SAVE] {name}: {len(split)} rows -> {out_path}")

    print("\nData prep done. Review the warnings above before running train.py.")


if __name__ == "__main__":
    main()
