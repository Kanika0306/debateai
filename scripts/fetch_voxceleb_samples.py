"""
fetch_voxceleb_samples.py - Extract actual VoxCeleb audio samples from VGGVox repo.
"""
import os
import shutil
import json
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VGGVOX_DIR = os.path.join(BASE, "scratch", "vggvox")
DEST_DIR = os.path.join(BASE, "data", "raw", "diarization", "voxceleb_sample")

def main():
    print("=" * 60)
    print("  Extracting VoxCeleb Audio Samples from VGGVox")
    print("=" * 60)

    if not os.path.exists(VGGVOX_DIR):
        print(f"Error: VGGVox repository not found at {VGGVOX_DIR}")
        return

    os.makedirs(DEST_DIR, exist_ok=True)

    # Locate sample files in VGGVox
    samples = [
        {
            "src": os.path.join(VGGVOX_DIR, "testfiles", "ident", "Y8hIVOBuels_0000002.wav"),
            "dest_name": "sample_001_ident.wav",
            "type": "identification",
            "youtube_id": "Y8hIVOBuels",
        },
        {
            "src": os.path.join(VGGVOX_DIR, "testfiles", "verif", "8jEAjG6SegY_0000008.wav"),
            "dest_name": "sample_002_verif.wav",
            "type": "verification",
            "youtube_id": "8jEAjG6SegY",
        },
        {
            "src": os.path.join(VGGVOX_DIR, "testfiles", "verif", "x6uYqmx31kE_0000001.wav"),
            "dest_name": "sample_003_verif.wav",
            "type": "verification",
            "youtube_id": "x6uYqmx31kE",
        }
    ]

    metadata = []

    for s in samples:
        src_path = s["src"]
        dest_path = os.path.join(DEST_DIR, s["dest_name"])

        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            size = os.path.getsize(dest_path)
            print(f"  [OK] Copied {s['dest_name']} ({size} bytes)")
            
            metadata.append({
                "filename": s["dest_name"],
                "type": s["type"],
                "youtube_id": s["youtube_id"],
                "size_bytes": size,
                "extracted_from": "https://github.com/a-nagrani/VGGVox",
                "extracted_at": datetime.now(timezone.utc).isoformat()
            })
        else:
            print(f"  [FAIL] Source file not found: {src_path}")

    # Write metadata.json
    meta_path = os.path.join(DEST_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [OK] Generated metadata.json at {meta_path}")

    # Write README.txt
    readme_path = os.path.join(DEST_DIR, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("VoxCeleb Dataset Sample\n")
        f.write("========================\n\n")
        f.write(f"Extracted: {datetime.now(timezone.utc).isoformat()}\n")
        f.write("Source: https://github.com/a-nagrani/VGGVox (testfiles/)\n")
        f.write("Purpose: Small sample audio files for speaker diarization testing.\n\n")
        f.write("Files:\n")
        for m in metadata:
            f.write(f"  - {m['filename']}: {m['type']} sample, Youtube ID {m['youtube_id']}, size {m['size_bytes']} bytes\n")
        
    print(f"  [OK] Generated README.txt at {readme_path}")
    print("\nExtraction complete.")

if __name__ == "__main__":
    main()
