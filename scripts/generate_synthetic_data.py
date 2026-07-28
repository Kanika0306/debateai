"""
generate_synthetic_data.py
============================
Synthetic data generator for DebateAI — Phase 0 gap-filling.

Targets (per project decision, July 2026):
  1. rag_chunks  -> expand gov_chunks.jsonl and worldbank_chunks.jsonl
  2. eval_set    -> expand data/processed/eval_set.jsonl with full synthetic
                    debate transcripts + gold labels (claims, verdicts, fallacies)
  3. fallacies   -> generate additional examples for underrepresented fallacy
                    classes in fallacy_examples_unified.jsonl

Supports Anthropic (claude-sonnet-4-6), Groq (llama-3.3-70b-versatile), or OpenAI (gpt-4o).

USAGE:
  python generate_synthetic_data.py rag_chunks --domain gov --n 150
  python generate_synthetic_data.py rag_chunks --domain worldbank --n 100
  python generate_synthetic_data.py eval_set --n 60
  python generate_synthetic_data.py fallacies --analyze
  python generate_synthetic_data.py fallacies --n 40 --classes "equivocation, hasty generalization, irrelevant authority, circular reasoning, red herring"
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUT_DIR = Path("./synthetic_out")
OUT_DIR.mkdir(exist_ok=True)


def get_llm_client_and_model():
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    # If key passed in ANTHROPIC_API_KEY starts with gsk_, route to Groq
    if anthropic_key and anthropic_key.startswith("gsk_"):
        groq_key = anthropic_key
        anthropic_key = None

    if groq_key or (anthropic_key and anthropic_key.startswith("gsk_")):
        try:
            import openai
            key = groq_key or anthropic_key
            client = openai.OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=key,
            )
            return "groq", client, "llama-3.3-70b-versatile"
        except ImportError:
            print("[ERROR] openai package required for Groq API. Run: pip install openai")
            sys.exit(1)

    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            return "anthropic", client, "claude-sonnet-4-6"
        except ImportError:
            print("[ERROR] anthropic package required. Run: pip install anthropic")
            sys.exit(1)

    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            return "openai", client, "gpt-4o"
        except ImportError:
            print("[ERROR] openai package required. Run: pip install openai")
            sys.exit(1)

    print("[ERROR] No valid API key found in ANTHROPIC_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY.")
    sys.exit(1)


def call_llm(prompt, max_tokens=4000):
    provider, client, model = get_llm_client_and_model()

    if provider == "anthropic":
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    else:  # groq or openai
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return resp.choices[0].message.content


def extract_json_array(text):
    """Strip markdown fences and parse a JSON array from model output."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
        else:
            text = parts[1]

    text = text.strip()
    # Find the start of [ and end of ]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    return json.loads(text)


# ---------------------------------------------------------------------------
# 1. RAG CHUNKS
# ---------------------------------------------------------------------------

RAG_DOMAIN_CONTEXT = {
    "gov": {
        "topics": [
            "public health policy", "agricultural subsidies", "education funding",
            "employment and labor statistics", "environmental regulation",
            "infrastructure spending", "tax policy", "social welfare programs",
            "digital governance / e-governance initiatives", "rural development schemes",
        ],
        "style": "Indian government open-data style (data.gov.in): factual, "
                 "policy-summary tone, includes specific figures/dates/scheme names.",
    },
    "worldbank": {
        "topics": [
            "global GDP growth", "poverty rate trends", "inflation and monetary policy",
            "employment and labor markets", "trade and exports", "public debt",
            "income inequality", "financial inclusion", "energy access",
            "urbanization trends",
        ],
        "style": "World Bank data-report style: economic indicators, "
                 "country/region comparisons, specific statistics with years.",
    },
}


def generate_rag_chunks(domain, n, chunk_size_words=180):
    cfg = RAG_DOMAIN_CONTEXT[domain]
    all_chunks = []
    batch_size = 10  # generate 10 chunks per API call
    n_batches = (n + batch_size - 1) // batch_size

    for b in range(n_batches):
        this_n = min(batch_size, n - len(all_chunks))
        topic = cfg["topics"][b % len(cfg["topics"])]
        prompt = f"""Generate {this_n} distinct, factually plausible knowledge-base
text chunks for a RAG (retrieval-augmented generation) fact-checking system.

Domain: {domain}
Sub-topic focus: {topic}
Style: {cfg['style']}

Each chunk should:
- Be ~{chunk_size_words} words
- Contain concrete, checkable facts (numbers, dates, named programs/reports)
  that a debate fact-checking claim might reference
- Be self-contained (a reader shouldn't need other chunks for context)
- Sound like real scraped web content, not a Q&A or listicle

Return ONLY a JSON array, no markdown fences, no commentary. Each element:
{{
  "text": "<the chunk text>",
  "source": "{domain}",
  "title": "<short descriptive title>",
  "topic": "{topic}"
}}
"""
        raw = call_llm(prompt, max_tokens=4000)
        try:
            batch = extract_json_array(raw)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [warn] batch {b+1} failed to parse ({e}), skipping")
            continue
        all_chunks.extend(batch)
        print(f"  batch {b+1}/{n_batches}: +{len(batch)} chunks (total {len(all_chunks)})")
        time.sleep(0.5)

    out_path = OUT_DIR / f"{domain}_chunks_synthetic.jsonl"
    now_iso = datetime.now(timezone.utc).isoformat()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(all_chunks):
            text_str = c.get("text", "")
            record = {
                "chunk_id": f"{domain}_synth_{i:04d}",
                "text": text_str,
                "source_url": f"https://synthetic.{domain}.org/data/{i:04d}",
                "title": c.get("title", ""),
                "fetch_date": now_iso,
                "trust_tier": 1,
                "domain_topic": c.get("topic", domain),
                "token_count": len(text_str.split()),
                "synthetic": True,
            }
            f.write(json.dumps(record) + "\n")
    print(f"\nWrote {len(all_chunks)} synthetic chunks -> {out_path}")
    print("Review a sample before merging into data/processed/rag_chunks/ "
          "and re-running scripts/build_rag_index.py")


# ---------------------------------------------------------------------------
# 2. EVAL SET (full debate transcripts with gold labels)
# ---------------------------------------------------------------------------

EVAL_TOPICS = [
    "climate change policy", "universal healthcare", "minimum wage increases",
    "immigration reform", "AI regulation", "renewable energy subsidies",
    "gun control legislation", "free college tuition", "trade tariffs",
    "criminal justice reform", "social media regulation", "nuclear energy expansion",
]

FALLACY_TYPES = [
    "Ad Hominem", "Strawman", "Red Herring", "Appeal to Emotion",
    "False Dilemma", "Slippery Slope", "Circular Reasoning", "Hasty Generalization",
    "None",  # no fallacy present
]


def generate_eval_set(n):
    records = []
    batch_size = 3  # full transcripts are long; small batches
    n_batches = (n + batch_size - 1) // batch_size

    for b in range(n_batches):
        this_n = min(batch_size, n - len(records))
        topic = EVAL_TOPICS[b % len(EVAL_TOPICS)]
        prompt = f"""Generate {this_n} synthetic two-speaker debate transcript
segments for evaluating a fact-checking + fallacy-detection AI pipeline.

Debate topic: {topic}

For EACH transcript segment, produce:
- A short exchange (3-5 sentences) between "Speaker A" and "Speaker B"
- At least one checkable factual claim embedded naturally in the dialogue
- Optionally, one logical fallacy embedded naturally (not all segments need one)
- Gold labels: the exact claim text extracted, a verdict (SUPPORTED / REFUTED /
  NOT_ENOUGH_INFO — pick what's realistic, don't make everything SUPPORTED),
  a one-sentence justification, and the fallacy type if present (from this list
  only: {', '.join(FALLACY_TYPES)})

Return ONLY a JSON array, no markdown fences, no commentary. Each element:
{{
  "transcript": "Speaker A: ...\\nSpeaker B: ...",
  "topic": "{topic}",
  "gold_claims": [
    {{"claim_text": "...", "speaker": "Speaker A", "verdict": "SUPPORTED",
      "justification": "..."}}
  ],
  "gold_fallacy": {{"type": "Strawman", "speaker": "Speaker B", "span": "..."}},
  "gold_fallacy_present": true
}}
If no fallacy present, set "gold_fallacy": null and "gold_fallacy_present": false.
"""
        raw = call_llm(prompt, max_tokens=4000)
        try:
            batch = extract_json_array(raw)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [warn] batch {b+1} failed to parse ({e}), skipping")
            continue
        records.extend(batch)
        print(f"  batch {b+1}/{n_batches}: +{len(batch)} transcripts (total {len(records)})")
        time.sleep(0.5)

    out_path = OUT_DIR / "eval_set_synthetic.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            r["id"] = f"synth_{i:04d}"
            r["synthetic"] = True
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(records)} synthetic eval transcripts -> {out_path}")
    print("IMPORTANT: spot-check verdicts manually — LLM-generated 'gold' labels "
          "are a starting point, not ground truth. Recommend hand-reviewing "
          "at least 20% before trusting this as a benchmark.")


# ---------------------------------------------------------------------------
# 3. FALLACY CLASS BALANCING
# ---------------------------------------------------------------------------

def analyze_fallacy_file(path):
    """Report class distribution of the existing unified fallacy file, if present locally."""
    if not os.path.exists(path):
        print(f"Could not find {path} locally to analyze. "
              f"Point --path to your actual fallacy_examples_unified.jsonl.")
        return None
    from collections import Counter
    counts = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                label = rec.get("label") or rec.get("fallacy_type") or rec.get("class")
                counts[label] += 1
            except json.JSONDecodeError:
                continue
    print("Class distribution:")
    for label, c in counts.most_common():
        print(f"  {label}: {c}")
    return counts


def generate_fallacy_examples(classes, n_per_class):
    all_examples = []
    for cls in classes:
        prompt = f"""Generate {n_per_class} distinct, realistic examples of the
"{cls}" logical fallacy, as they would naturally appear in a spoken political
or public debate (not textbook-style examples).

Each example should:
- Be 1-3 sentences, sound like something a real debater would say out loud
- Clearly exhibit the "{cls}" fallacy without being a caricature
- Vary in topic (mix politics, economics, health, technology, environment)

Return ONLY a JSON array, no markdown fences, no commentary. Each element:
{{"text": "...", "label": "{cls}", "topic": "..."}}
"""
        raw = call_llm(prompt, max_tokens=3000)
        try:
            batch = extract_json_array(raw)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [warn] class '{cls}' failed to parse ({e}), skipping")
            continue
        for ex in batch:
            ex["fallacy_type"] = cls
            ex["label"] = cls
            ex["source"] = "synthetic"
        all_examples.extend(batch)
        print(f"  {cls}: +{len(batch)} examples")
        time.sleep(0.5)

    out_path = OUT_DIR / "fallacy_examples_synthetic.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(all_examples):
            ex["id"] = f"synth_fallacy_{i:04d}"
            ex["synthetic"] = True
            f.write(json.dumps(ex) + "\n")
    print(f"\nWrote {len(all_examples)} synthetic fallacy examples -> {out_path}")
    print("Merge into fallacy_examples_unified.jsonl, then re-check class "
          "balance with --analyze before training.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DebateAI synthetic data generator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_rag = sub.add_parser("rag_chunks")
    p_rag.add_argument("--domain", choices=["gov", "worldbank"], required=True)
    p_rag.add_argument("--n", type=int, default=100)

    p_eval = sub.add_parser("eval_set")
    p_eval.add_argument("--n", type=int, default=60)

    p_fal = sub.add_parser("fallacies")
    p_fal.add_argument("--analyze", action="store_true")
    p_fal.add_argument("--path", default="data/processed/fallacies/fallacy_examples_unified.jsonl")
    p_fal.add_argument("--n", type=int, default=40, help="examples per class")
    p_fal.add_argument("--classes", default="Slippery Slope,False Dilemma,Circular Reasoning")

    args = parser.parse_args()

    if args.command == "rag_chunks":
        generate_rag_chunks(args.domain, args.n)
    elif args.command == "eval_set":
        generate_eval_set(args.n)
    elif args.command == "fallacies":
        if args.analyze:
            analyze_fallacy_file(args.path)
        else:
            classes = [c.strip() for c in args.classes.split(",")]
            generate_fallacy_examples(classes, args.n)


if __name__ == "__main__":
    main()
