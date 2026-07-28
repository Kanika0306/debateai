"""
generate_voice_data.py
========================
Synthetic voice generator for DebateAI evaluation transcripts.
Synthesizes multi-speaker audio from data/processed/eval_set.jsonl using Microsoft Edge TTS.

Requires: pip install edge-tts pydub
Requires: ffmpeg installed on PATH (auto-detects WinGet installation)

USAGE:
  python generate_voice_data.py --dry-run
  python generate_voice_data.py --limit 3
  python generate_voice_data.py
"""

import argparse
import asyncio
import glob
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# Auto-detect FFmpeg from WinGet or common locations if not on PATH
def setup_ffmpeg():
    winget_matches = glob.glob(
        r"C:\Users\kanik\AppData\Local\Microsoft\WinGet\Packages\**\ffmpeg.exe",
        recursive=True,
    )
    if winget_matches:
        bin_dir = os.path.dirname(winget_matches[0])
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        try:
            from pydub import AudioSegment
            AudioSegment.converter = os.path.join(bin_dir, "ffmpeg.exe")
            AudioSegment.ffprobe = os.path.join(bin_dir, "ffprobe.exe")
        except ImportError:
            pass

setup_ffmpeg()

ROOT = Path(__file__).resolve().parent
EVAL_SET_PATH = ROOT / "data" / "processed" / "eval_set.jsonl"
OUT_AUDIO_DIR = ROOT / "data" / "processed" / "audio"
METADATA_OUT_PATH = ROOT / "data" / "processed" / "audio" / "audio_metadata.jsonl"

SPEAKER_VOICES = {
    "Speaker A": "en-US-GuyNeural",
    "Speaker B": "en-US-JennyNeural",
    "Speaker C": "en-US-ChristopherNeural",
    "Speaker D": "en-US-AriaNeural",
    "DEFAULT_A": "en-US-GuyNeural",
    "DEFAULT_B": "en-US-JennyNeural",
}


def parse_turns(record):
    """
    Parse dialogue transcript into a list of (speaker, text) tuples.
    Fallback to constructing a claim-evidence exchange if transcript field is missing.
    """
    transcript = record.get("transcript", "")
    turns = []

    if transcript:
        normalized = transcript.replace(" / ", "\n")
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]

        for line in lines:
            match = re.match(r"^(Speaker\s+[A-Z0-9]+):\s*(.*)$", line, re.IGNORECASE)
            if match:
                spk = match.group(1).title()
                txt = match.group(2).strip()
                turns.append((spk, txt))
            else:
                if turns:
                    prev_spk, prev_txt = turns[-1]
                    turns[-1] = (prev_spk, f"{prev_txt} {line}")
                else:
                    turns.append(("Speaker A", line))

    if not turns:
        claim = record.get("claim")
        evidence = record.get("evidence")
        if claim:
            turns.append(("Speaker A", f"Claim: {claim}"))
            if evidence:
                turns.append(("Speaker B", f"Evidence: {evidence}"))

    return turns


async def synth_turn(text, voice, out_path):
    import edge_tts
    comm = edge_tts.Communicate(text, voice)
    await comm.save(str(out_path))


def combine_audio_files(turn_files, final_path):
    """Combine multiple turn mp3 files with pause using pydub."""
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    pause = AudioSegment.silent(duration=400)  # 400ms pause between turns

    for idx, fpath in enumerate(turn_files):
        seg = AudioSegment.from_file(str(fpath))
        if idx > 0:
            combined += pause
        combined += seg

    final_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(final_path), format="mp3")


async def process_record(record, idx_default, out_dir, dry_run=False):
    rec_id = record.get("id", f"eval_{idx_default:04d}")
    topic = record.get("topic", "claim_verification")

    turns = parse_turns(record)
    if not turns:
        print(f"  [warn] No turns parsed for record {rec_id}")
        return None

    if dry_run:
        print(f"\n[DRY RUN] Record: {rec_id} (Topic: {topic})")
        for spk, txt in turns:
            voice = SPEAKER_VOICES.get(spk, SPEAKER_VOICES["DEFAULT_A"] if "A" in spk else SPEAKER_VOICES["DEFAULT_B"])
            print(f"  - [{spk} -> {voice}]: {txt[:70]}...")
        return {
            "id": rec_id,
            "topic": topic,
            "turn_count": len(turns),
            "dry_run": True,
        }

    final_audio_path = out_dir / f"{rec_id}.mp3"
    turn_files = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for idx, (spk, txt) in enumerate(turns):
            voice = SPEAKER_VOICES.get(
                spk,
                SPEAKER_VOICES["DEFAULT_A"] if "A" in spk or idx % 2 == 0 else SPEAKER_VOICES["DEFAULT_B"],
            )
            t_file = tmp_path / f"turn_{idx:02d}.mp3"
            await synth_turn(txt, voice, t_file)
            turn_files.append(t_file)

        combine_audio_files(turn_files, final_audio_path)

    print(f"\n  Synthesized {rec_id} -> {final_audio_path} ({len(turns)} turns)", flush=True)
    return {
        "id": rec_id,
        "topic": topic,
        "audio_file": f"{rec_id}.mp3",
        "turn_count": len(turns),
        "transcript": record.get("transcript") or f"Speaker A: {record.get('claim')} / Speaker B: {record.get('evidence')}",
    }


def main():
    parser = argparse.ArgumentParser(description="DebateAI Voice Data Generator")
    parser.add_argument("--dry-run", action="store_true", help="Print synthesis plan without generating audio")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to process")
    parser.add_argument("--eval-file", default=str(EVAL_SET_PATH), help="Path to eval_set.jsonl")
    parser.add_argument("--out-dir", default=str(OUT_AUDIO_DIR), help="Output directory for audio files")

    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        print(f"[ERROR] Eval set file not found at {eval_path}")
        sys.exit(1)

    records = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if args.limit:
        records = records[: args.limit]

    print(f"Loaded {len(records)} records from {eval_path}")
    if args.dry_run:
        print("Running in DRY-RUN mode (no audio will be written)...")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for i, r in enumerate(records, 1):
        print(f"Processing {i}/{len(records)}...", end="", flush=True)
        res = asyncio.run(process_record(r, i - 1, out_dir, dry_run=args.dry_run))
        if res:
            metadata.append(res)

    if not args.dry_run:
        with open(METADATA_OUT_PATH, "w", encoding="utf-8") as f:
            for m in metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"\nWrote audio metadata for {len(metadata)} files -> {METADATA_OUT_PATH}")

    print("\nDone!")


if __name__ == "__main__":
    main()
