"""
generate_synthetic_fallacies.py
================================
Generates synthetic training examples for the two remaining weak fallacy
classes (appeal to emotion: 0.45 F1, fallacy of extension: 0.38 F1)
using the Groq API (fast + free-tier friendly).
"""

import argparse
import json
import os
import time
import re
from pathlib import Path
import dotenv
dotenv.load_dotenv()

import openai

MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]

CLASS_DEFS = {
    "appeal to emotion": {
        "definition": (
            "The argument tries to win agreement by triggering fear, pity, "
            "anger, pride, or guilt in the listener, INSTEAD of offering "
            "evidence or logical reasoning. The emotional appeal replaces "
            "the argument rather than supporting it."
        ),
        "avoid": (
            "Do NOT write ad hominem (attacking the person), do NOT write "
            "appeal to fear framed as a personal attack, and do NOT include "
            "any real facts/statistics that would make it a legitimate "
            "evidence-based claim."
        ),
        "examples": [
            "Think of the children who will suffer if we don't pass this law immediately.",
            "How can you support this policy when it will break the hearts of thousands of families?",
        ],
    },
    "fallacy of extension": {
        "definition": (
            "The argument distorts, exaggerates, or extends an opponent's "
            "actual position into a more extreme version, then attacks that "
            "extreme version instead of what was really said (i.e. a "
            "strawman). The exaggeration must clearly go beyond the original claim."
        ),
        "avoid": (
            "Do NOT simply restate the opponent's real position accurately. "
            "The example MUST show a visible distortion/exaggeration of a "
            "stated or implied original position."
        ),
        "examples": [
            "So you're saying we should just let anyone walk across the border with no rules at all?",
            "If you support raising the minimum wage a little, you must want businesses to go completely bankrupt.",
        ],
    },
}

PROMPT_TEMPLATE = """You are generating training data for a fallacy-detection classifier.

Fallacy class: {name}
Definition: {definition}
Constraints: {avoid}

Example style (do not repeat these, just match the style/length):
- "{ex1}"
- "{ex2}"

Generate {n} NEW, DIVERSE, single-sentence or short (1-2 sentence) statements
that clearly exhibit this fallacy. Vary the topic across politics, sports,
health, technology, relationships, business, education, and everyday debates.
Do not number them or add commentary.

Return ONLY a JSON array of strings, nothing else. Example format:
["statement one", "statement two", "statement three"]
"""


def call_groq(client, class_name, defn, n_batch):
    prompt = PROMPT_TEMPLATE.format(
        name=class_name,
        definition=defn["definition"],
        avoid=defn["avoid"],
        ex1=defn["examples"][0],
        ex2=defn["examples"][1],
        n=n_batch,
    )
    
    text = ""
    for model_name in MODELS:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=2000,
            )
            text = resp.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f"  [warn] Model {model_name} rate limited / failed: {e}. Trying fallback model...")
            time.sleep(1)

    if not text:
        return []

    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        items = json.loads(text)
        return [s.strip() for s in items if isinstance(s, str) and len(s.strip()) > 10]
    except json.JSONDecodeError:
        print(f"  [warn] failed to parse batch for {class_name}, skipping")
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--output", default="data/processed/fallacies/synthetic_topup.jsonl")
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY environment variable in .env first.")
    
    client = openai.OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    rows = []
    seen = set()

    for class_name, defn in CLASS_DEFS.items():
        print(f"\n[generating] {class_name} — target {args.per_class}")
        collected = 0
        attempts = 0
        while collected < args.per_class and attempts < 40:
            attempts += 1
            n_batch = min(args.batch_size, args.per_class - collected)
            batch = call_groq(client, class_name, defn, n_batch)
            for s in batch:
                key = s.lower().strip()
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "text": s,
                    "fallacy_type": class_name,
                    "source": "synthetic_groq",
                    "synthetic": True,
                })
                collected += 1
                if collected >= args.per_class:
                    break
            print(f"  {collected}/{args.per_class} collected (attempt {attempts})")
            time.sleep(0.5)

        if collected < args.per_class:
            print(f"  [warn] only got {collected}/{args.per_class} for {class_name}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(rows)} synthetic rows -> {out_path}")

if __name__ == "__main__":
    main()
