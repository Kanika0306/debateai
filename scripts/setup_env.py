"""
setup_env.py — Create the full directory structure for debate-ai raw data layer.
"""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIRS = [
    "data/raw/fact_verification/fever",
    "data/raw/fact_verification/liar",
    "data/raw/fact_verification/feverous",
    "data/raw/claim_detection/claimbuster",
    "data/raw/fallacies/argotario",
    "data/raw/fallacies/logic_dataset",
    "data/raw/misinformation/fakenewsnet",
    "data/raw/rag_sources/gov",
    "data/raw/rag_sources/who",
    "data/raw/rag_sources/worldbank",
    "data/raw/rag_sources/nasa_climate",
    "data/raw/rag_sources/semantic_scholar",
    "data/raw/speech/common_voice_sample",
    "data/raw/diarization/voxceleb_sample",
    "data/raw/diarization/ami_sample",
    "data/processed",
    "scripts",
]

def main():
    for d in DIRS:
        full = os.path.join(BASE, d)
        os.makedirs(full, exist_ok=True)
        print(f"  [OK] {d}")
    print(f"\nAll {len(DIRS)} directories created under {BASE}")

if __name__ == "__main__":
    main()
