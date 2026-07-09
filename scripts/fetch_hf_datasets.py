"""
fetch_hf_datasets.py - Download HuggingFace datasets for debate-ai.
Saves each dataset as parquet + sample JSON + README.
"""
import os
import sys
import json

# Force UTF-8 output on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import traceback
from datetime import datetime, timezone

# -- paths --
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE, "data", "raw")

# -- dataset definitions --
DATASETS = [
    {
        "name": "FEVER",
        "load_args": [
            {"path": "fever/fever", "name": "v1.0"},
            {"path": "fever/fever"},
            # Community mirrors / alternatives if official uses loading script
            {"path": "pietrolesci/fever"},
            {"path": "copenlu/fever_gold_evidence"},
        ],
        "dest": os.path.join(DATA_RAW, "fact_verification", "fever"),
    },
    {
        "name": "LIAR",
        "load_args": [
            {"path": "ucsbai/liar"},
            {"path": "chengxuphd/liar2"},
            {"path": "ucsbnlp/liar"},
        ],
        "dest": os.path.join(DATA_RAW, "fact_verification", "liar"),
    },
    {
        "name": "FEVEROUS",
        "load_args": [
            {"path": "fever/feverous"},
            # Try loading parquet directly from the HF hub if script-based loading fails
            {"path": "parquet", "data_files": "hf://datasets/fever/feverous@refs%2Fconvert%2Fparquet/default/train/0000.parquet"},
        ],
        "dest": os.path.join(DATA_RAW, "fact_verification", "feverous"),
    },
    {
        "name": "Argotario",
        "load_args": [
            {"path": "argotario"},
            # Argotario is included in the FADES aggregated dataset
            {"path": "maxime-antoine-dev/fades-dataset"},
        ],
        "dest": os.path.join(DATA_RAW, "fallacies", "argotario"),
    },
    {
        "name": "Logic Dataset (logical_fallacy)",
        "load_args": [
            # Note: hyphen not underscore
            {"path": "tasksource/logical-fallacy"},
            {"path": "tasksource/logical_fallacy"},
        ],
        "dest": os.path.join(DATA_RAW, "fallacies", "logic_dataset"),
    },
]


def save_dataset(ds_dict, dest, ds_name):
    """Save a DatasetDict to parquet + sample + README."""
    os.makedirs(dest, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    split_info = {}
    columns = None

    for split_name, split_ds in ds_dict.items():
        num_rows = len(split_ds)
        split_info[split_name] = num_rows
        if columns is None:
            columns = split_ds.column_names

        # save parquet
        parquet_path = os.path.join(dest, f"{split_name}.parquet")
        split_ds.to_parquet(parquet_path)
        print(f"    Saved {split_name}.parquet ({num_rows} rows)")

    # save sample
    first_split = list(ds_dict.keys())[0]
    sample = ds_dict[first_split].select(range(min(5, len(ds_dict[first_split]))))
    sample_path = os.path.join(dest, "sample_5_rows.json")
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump([dict(row) for row in sample], f, indent=2, default=str)
    print(f"    Saved sample_5_rows.json")

    # save README
    readme_path = os.path.join(dest, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {ds_name}\n")
        f.write(f"Downloaded: {now}\n")
        f.write(f"Columns: {columns}\n\n")
        f.write("Splits:\n")
        for s, n in split_info.items():
            f.write(f"  {s}: {n} rows\n")
    print(f"    Saved README.txt")

    return split_info, columns


def main():
    from datasets import load_dataset, DatasetDict

    results = []

    for entry in DATASETS:
        name = entry["name"]
        dest = entry["dest"]
        print(f"\n{'='*60}")
        print(f"  Loading: {name}")
        print(f"{'='*60}")

        loaded = False
        last_error = None

        for args in entry["load_args"]:
            try:
                path = args["path"]
                config = args.get("name")
                data_files = args.get("data_files")
                print(f"  Trying: path={path}, config={config}, data_files={data_files}")
                if data_files:
                    ds = load_dataset(path, data_files=data_files)
                elif config:
                    ds = load_dataset(path, config)
                else:
                    ds = load_dataset(path)

                # Wrap single Dataset in a DatasetDict
                if not isinstance(ds, DatasetDict):
                    ds = DatasetDict({"train": ds})

                split_info, columns = save_dataset(ds, dest, name)
                print(f"\n  [OK] {name} -- Splits: {split_info}")
                print(f"    Columns: {columns}")
                results.append({
                    "name": name,
                    "status": "success",
                    "splits": split_info,
                    "columns": columns,
                    "path": dest,
                    "repo_id": path,
                })
                loaded = True
                break
            except Exception as e:
                last_error = str(e)
                print(f"  [FAIL] Failed with {path}: {last_error}")
                traceback.print_exc()

        if not loaded:
            print(f"\n  [FAIL] {name} -- ALL attempts failed. Last error: {last_error}")
            results.append({
                "name": name,
                "status": "failed",
                "error": last_error,
                "path": dest,
            })

    # ── Summary ──
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Name':<35} {'Status':<12} {'Rows':<20} {'Repo ID'}")
    print("-" * 90)
    for r in results:
        rows = str(r.get("splits", "")) if r["status"] == "success" else r.get("error", "")[:40]
        repo = r.get("repo_id", "N/A")
        print(f"{r['name']:<35} {r['status']:<12} {rows:<20} {repo}")

    # Save results JSON
    results_path = os.path.join(DATA_RAW, "hf_download_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
