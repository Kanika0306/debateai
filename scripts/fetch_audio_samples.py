"""
fetch_audio_samples.py - Fetch audio samples for debate-ai.
- Common Voice: Download 50 test samples (requires HF token + terms accepted)
- VoxCeleb: Print manual download instructions
- AMI: Print manual download instructions
"""
import os
import sys
import json

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE, "data", "raw")

results = []


def fetch_common_voice():
    """Download 50 Common Voice test samples."""
    print(f"\n{'='*60}")
    print("  Common Voice (50 test samples)")
    print(f"{'='*60}")

    dest = os.path.join(DATA_RAW, "speech", "common_voice_sample")
    os.makedirs(dest, exist_ok=True)

    try:
        from datasets import load_dataset
        import soundfile as sf

        print("  Loading mozilla-foundation/common_voice_17_0 (en, test[:50])...")
        print("  NOTE: Requires HF token + accepted terms on huggingface.co")

        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "en",
            split="test[:50]",
            
        )

        print(f"  Loaded {len(ds)} samples")
        print(f"  Columns: {ds.column_names}")

        # Save audio files + metadata
        metadata = []
        for i, row in enumerate(ds):
            audio = row.get("audio", {})
            if audio and "array" in audio:
                audio_path = os.path.join(dest, f"sample_{i:03d}.wav")
                sf.write(audio_path, audio["array"], audio["sampling_rate"])

            meta = {k: v for k, v in row.items() if k != "audio"}
            meta["audio_file"] = f"sample_{i:03d}.wav"
            metadata.append(meta)

        # Save metadata
        meta_path = os.path.join(dest, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        # README
        readme_path = os.path.join(dest, "README.txt")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"Common Voice 17.0 (English) - Test Split Sample\n")
            f.write(f"Downloaded: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"Samples: {len(ds)}\n")
            f.write(f"Columns: {ds.column_names}\n")
            f.write(f"Source: mozilla-foundation/common_voice_17_0\n")

        print(f"  [OK] Saved {len(ds)} audio samples to {dest}")
        results.append({
            "name": "Common Voice",
            "status": "success",
            "samples": len(ds),
            "path": dest,
        })

    except ImportError:
        print("  [FAIL] soundfile not installed. Run: pip install soundfile")
        results.append({"name": "Common Voice", "status": "failed (soundfile not installed)"})
    except Exception as e:
        error_msg = str(e)
        print(f"  [FAIL] Failed: {error_msg}")

        if "gated" in error_msg.lower() or "token" in error_msg.lower() or "401" in error_msg:
            print("\n  To fix this:")
            print("  1. Go to https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
            print("  2. Accept the dataset terms/license")
            print("  3. Set HF_TOKEN env var or run: huggingface-cli login")

        results.append({
            "name": "Common Voice",
            "status": f"failed: {error_msg[:100]}",
            "path": dest,
        })


def write_voxceleb_instructions():
    """Write manual download instructions for VoxCeleb."""
    print(f"\n{'='*60}")
    print("  VoxCeleb (Manual Download Required)")
    print(f"{'='*60}")

    dest = os.path.join(DATA_RAW, "diarization", "voxceleb_sample")
    os.makedirs(dest, exist_ok=True)

    instructions = """VoxCeleb Dataset - Manual Download Instructions
================================================

VoxCeleb requires manual license agreement before download.

Steps:
1. Visit: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
2. Read and accept the license agreement
3. Register for access (academic email may be required)
4. Download VoxCeleb1 (for speaker verification):
   - vox1_dev_wav.zip (training set)
   - vox1_test_wav.zip (test set)
5. For VoxCeleb2 (larger dataset for diarization):
   - vox2_dev_aac.zip
   - vox2_test_aac.zip

After downloading:
- Extract to this directory: data/raw/diarization/voxceleb_sample/
- For a small sample, extract only a few speakers (e.g., 5-10)

Citation:
  Nagrani, A., Chung, J. S., & Zisserman, A. (2017).
  VoxCeleb: A Large-Scale Speaker Identification Dataset.
  INTERSPEECH.

Note: For debate-ai, you only need a small sample for testing
speaker diarization. Consider downloading just the test set.
"""
    path = os.path.join(dest, "DOWNLOAD_INSTRUCTIONS.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(instructions)

    print(f"  Instructions saved to {path}")
    print("  -> Visit https://www.robots.ox.ac.uk/~vgg/data/voxceleb/")
    print("  -> Accept license and download manually")

    results.append({
        "name": "VoxCeleb",
        "status": "manual-required",
        "path": dest,
    })


def write_ami_instructions():
    """Write manual download instructions for AMI Meeting Corpus."""
    print(f"\n{'='*60}")
    print("  AMI Meeting Corpus (Manual Download Required)")
    print(f"{'='*60}")

    dest = os.path.join(DATA_RAW, "diarization", "ami_sample")
    os.makedirs(dest, exist_ok=True)

    instructions = """AMI Meeting Corpus - Manual Download Instructions
===================================================

The AMI Meeting Corpus requires registration and license agreement.

Steps:
1. Visit: https://groups.inf.ed.ac.uk/ami/corpus/
2. Read the license terms
3. Register for access
4. Download options:
   a. Individual headset mix (IHM) - best for diarization
   b. Single distant microphone (SDM)
   c. Multiple distant microphones (MDM)
5. For debate-ai testing, download just a few meetings:
   - ES2002a, ES2002b, ES2003a (example meeting IDs)
   - Download headset mix audio + annotations

Alternative via HuggingFace (may still need license):
  from datasets import load_dataset
  ds = load_dataset("edinburghcstr/ami", "ihm", split="test[:10]")

After downloading:
- Extract to: data/raw/diarization/ami_sample/
- Include both audio files and RTTM annotation files

Citation:
  Carletta, J. (2007).
  Unleashing the killer corpus: experiences in creating the
  multi-everything AMI Meeting Corpus.
  Language Resources and Evaluation.

Note: For debate-ai, a sample of 5-10 meetings is sufficient
for testing and developing the diarization pipeline.
"""
    path = os.path.join(dest, "DOWNLOAD_INSTRUCTIONS.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(instructions)

    print(f"  Instructions saved to {path}")
    print("  -> Visit https://groups.inf.ed.ac.uk/ami/corpus/")
    print("  -> Register and download manually")

    results.append({
        "name": "AMI Meeting Corpus",
        "status": "manual-required",
        "path": dest,
    })


def main():
    fetch_common_voice()
    write_voxceleb_instructions()
    write_ami_instructions()

    print(f"\n\n{'='*60}")
    print("  AUDIO SAMPLES SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['name']}: {r['status']}")

    results_path = os.path.join(DATA_RAW, "audio_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
