# generate_synthetic.py
import json, time, random, os
import dotenv
dotenv.load_dotenv()
import openai
import anthropic

WEAK_CLASSES = {
    "fallacy of logic": """A statement that violates basic logical structure — 
        the conclusion doesn't follow from premises even if premises are true.
        Example: 'All dogs are animals. Some animals are cats. Therefore some dogs are cats.'""",

    "fallacy of relevance": """An argument where the premise is irrelevant to the conclusion.
        Example: 'We should trust this politician because he coaches little league.'""",

    "red herring": """Introducing an irrelevant topic to distract from the real issue.
        Example: 'Why worry about my speeding ticket when people are dying in wars?'""",

    "intentional": """A deliberate, knowing misrepresentation or deceptive framing.
        Example: 'I never said she stole the money' (implying someone else said it).""",

    "equivocation": """Using a word with two different meanings in the same argument.
        Example: 'The sign said fine for parking here, so I parked there because it was fine.'""",

    "fallacy of credibility": """Dismissing or accepting a claim based solely on the speaker's 
        identity rather than the argument's merit.
        Example: 'You can't trust his economic analysis, he's a poet.'""",
}

TARGET_PER_CLASS = 300  # generates ~300 new examples per weak class

def get_client_and_provider():
    # 1. Groq (100% Free API)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        print("[info] Using Groq Free API (llama-3.3-70b)...")
        return openai.OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1"), "groq"

    # 2. OpenRouter (Free models)
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        print("[info] Using OpenRouter API...")
        return openai.OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1"), "openrouter"

    # 3. Grok (xAI)
    grok_key = os.environ.get("GROQ_API_KEY") or os.environ.get("XAI_API_KEY")
    if grok_key and grok_key.startswith("xai-"):
        print("[info] Using Grok (xAI) API...")
        return openai.OpenAI(api_key=grok_key, base_url="https://api.x.ai/v1"), "grok"

    # 4. Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("[info] Using Anthropic Claude API...")
        return anthropic.Anthropic(api_key=anthropic_key), "anthropic"

    # 5. OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and not openai_key.startswith("sk-proj-placeholder"):
        print("[info] Using OpenAI API...")
        return openai.OpenAI(api_key=openai_key), "openai"

    return None, None

def generate_examples(client, provider, fallacy_name, description, n=10):
    prompt = f"""Generate {n} short, distinct debate or political speech sentences 
that clearly demonstrate the '{fallacy_name}' logical fallacy.

Definition: {description}

Rules:
- Each sentence must be 1-3 sentences max
- Sound like real debate speech, not textbook examples  
- Cover diverse topics (politics, science, economics, social issues)
- Each must be clearly and unambiguously this specific fallacy
- Do NOT label them or add explanations

Return ONLY a JSON array of strings, no other text:
["example 1", "example 2", ...]"""

    if provider == "groq":
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()

    elif provider == "openrouter":
        resp = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()

    elif provider == "grok":
        try:
            resp = client.chat.completions.create(
                model="grok-2-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
        except Exception:
            resp = client.chat.completions.create(
                model="grok-beta",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
        text = resp.choices[0].message.content.strip()

    elif provider == "anthropic":
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception:
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
        text = response.content[0].text.strip()

    elif provider == "openai":
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()
    else:
        raise ValueError("No valid API provider available.")

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    
    return json.loads(text.strip())

def main():
    client, provider = get_client_and_provider()
    if not client:
        print("[error] No active free API key found in .env")
        print("Please add GROQ_API_KEY=gsk_... to your .env file.")
        print("Get a 100% free key at: https://console.groq.com/keys")
        return

    all_new = []

    for fallacy_name, description in WEAK_CLASSES.items():
        print(f"\nGenerating for: {fallacy_name}")
        examples = []
        batches = TARGET_PER_CLASS // 10
        
        for i in range(batches):
            try:
                batch = generate_examples(client, provider, fallacy_name, description, n=10)
                examples.extend(batch)
                print(f"  Batch {i+1}/{batches}: {len(batch)} examples")
                time.sleep(1.0)  # rate limit buffer
            except Exception as e:
                print(f"  Batch {i+1} failed: {e}")
                time.sleep(2)
        
        for text in examples:
            all_new.append({
                "text": text,
                "fallacy_type": fallacy_name,
                "source": f"synthetic_{provider}",
                "synthetic": True
            })
        
        print(f"  Total for {fallacy_name}: {len(examples)}")

    # Save
    import pandas as pd
    new_df = pd.DataFrame(all_new)
    print(f"\nTotal new synthetic examples: {len(new_df)}")
    if len(new_df) > 0:
        print(new_df['fallacy_type'].value_counts())

    # Merge with existing
    existing = pd.read_json(
        'data/processed/fallacies/fallacy_examples_unified.jsonl', lines=True
    )
    if len(new_df) > 0:
        combined = pd.concat([existing, new_df[['text','fallacy_type','source','synthetic']]], 
                             ignore_index=True)
        combined.to_json(
            'data/processed/fallacies/fallacy_examples_augmented.jsonl',
            orient='records', lines=True
        )
        print(f"Saved augmented dataset: {len(combined)} total rows")
    else:
        print("[warn] No new synthetic examples generated. Dataset was not updated.")

if __name__ == "__main__":
    main()
