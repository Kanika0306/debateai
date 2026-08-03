"""
judge_agent.py
==============
DebateAI Judge Agent.
Takes structured output from pipeline.py (veracity, fallacy, NLI)
and synthesizes a structured natural-language verdict using LLM API.
"""

import os
import sys
import json
import re
import argparse
import dotenv

dotenv.load_dotenv()

SYSTEM_PROMPT = """You are a debate analysis judge. You receive structured output from three specialist ML classifiers (veracity, fallacy detection, and optional fact-verification) about a single statement made in a debate. Your job is to turn these raw signals into a clear, fair, human-readable verdict.

CRITICAL RULES:
1. The veracity classifier was trained mainly on political/social claims. It is known to be unreliable on out-of-domain topics like science or history, and it can be confidently wrong. Treat "reliable": true as a weak signal, not proof. If the claim is about a topic you have strong independent knowledge of (e.g. established science, historical dates, basic facts) and the classifier's veracity label contradicts what you know to be true, say so explicitly and trust your own knowledge over the classifier — do not defer to a wrong confident label.
2. If "reliable": false, explicitly tell the user the veracity signal is low-confidence noise and should not be treated as a real judgment.
3. Only discuss fact-checking if fact_check is not null. If fact_check is null, do NOT claim the statement was fact-checked — say no evidence was provided for comparison, if relevant.
4. Only report a fallacy if its confidence is reasonably above the top3 runner-up (a large margin = real signal; a close spread across top3 = likely noise, say so rather than asserting a fallacy occurred).
5. Never invent evidence, sources, or facts not present in the input or in your own general knowledge — if uncertain, say you're uncertain.
6. Be fair and even-handed regardless of the statement's political content or which "side" it favors — you're judging argument quality and factual grounding, not taking a position.
7. Keep the verdict readable by a general audience — avoid ML jargon like "softmax" or "macro F1" in the user-facing summary field, but you may reference confidence levels in plain language ("the model was uncertain").
8. SILENT DISAGREEMENT IS NOT ALLOWED: If your assessment of a signal's trustworthiness differs from the "reliable" flag you were given (e.g. if the veracity classifier output has "reliable": true, but you choose not to trust it because it contradicts established factual knowledge or lacks context; or if "reliable": false but you trust it), you MUST explicitly state this disagreement in confidence_caveats (e.g. "Note: classifier marked this as reliable, but I am treating it with caution because X").

OUTPUT FORMAT — return ONLY valid JSON, no preamble, matching this schema:
{
  "verdict_summary": "1-3 sentence plain-language verdict",
  "veracity_assessment": {
    "trust_classifier": true/false,
    "reasoning": "string explanation"
  },
  "fallacy_assessment": {
    "fallacy_detected": true/false,
    "fallacy_name": "string or null",
    "explanation": "string explanation or null"
  },
  "fact_check_assessment": {
    "performed": true/false,
    "verdict": "string or null",
    "claim_is_supported": true/false/null,
    "explanation": "string explanation or null"
  },
  "confidence_caveats": ["list of strings"]
}"""

USER_TEMPLATE = """Statement: "{text}"
Speaker: {speaker}
Context: {context}

Veracity model output: label={v_label}, confidence={v_conf}, reliable={v_rel}
Fallacy model output: top prediction={f_label} (confidence={f_conf}), runner-ups={f_runnerups}
Fact-check: {fact_check_str}"""


def _clean_and_parse_json(text: str) -> dict:
    """Helper to parse JSON output from LLM, handling markdown fences and preambles."""
    cleaned = text.strip()
    # Strip markdown code blocks
    cleaned = re.sub(r"^```(json)?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()

    # Try direct JSON parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Extract JSON substring if surrounded by preamble or trailing prose
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Failed to parse LLM response as JSON:\n{text}")


def judge(pipeline_output: dict, speaker: str = None, context: str = None) -> dict:
    """
    Main Judge Agent function.
    Takes structured output from pipeline.py analyze_text() and generates a structured verdict.
    """
    text = pipeline_output.get("text", "")
    spk = speaker if speaker else "unknown"
    ctx = context if context else "none provided"

    veracity = pipeline_output.get("veracity", {}) or {}
    v_label = veracity.get("label", "unknown")
    v_conf = veracity.get("confidence", 0.0)
    v_rel = veracity.get("reliable", False)

    fallacy = pipeline_output.get("fallacy", {}) or {}
    f_label = fallacy.get("label", "no fallacy")
    f_conf = fallacy.get("confidence", 0.0)
    top3 = fallacy.get("top3", []) or []
    f_runnerups = top3[1:] if len(top3) > 1 else []

    fc = pipeline_output.get("fact_check")
    claim_is_supported = None
    if fc and fc.get("verdict"):
        raw_v = str(fc.get("verdict")).upper()
        if raw_v == "SUPPORTS":
            claim_is_supported = True
        elif raw_v == "REFUTES":
            claim_is_supported = False
        elif raw_v == "NEI":
            claim_is_supported = None

        fact_check_str = (
            f"verdict={fc.get('verdict')} (claim_is_supported={claim_is_supported}), "
            f"confidence={fc.get('confidence')}"
        )
    else:
        fact_check_str = "not performed — no evidence text was provided"

    user_msg = USER_TEMPLATE.format(
        text=text,
        speaker=spk,
        context=ctx,
        v_label=v_label,
        v_conf=v_conf,
        v_rel=v_rel,
        f_label=f_label,
        f_conf=f_conf,
        f_runnerups=f_runnerups,
        fact_check_str=fact_check_str,
    )

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    raw_response = ""

    # Primary: Anthropic API if key is present
    if anthropic_key:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        models_to_try = ["claude-sonnet-4-6", "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022"]
        for m in models_to_try:
            try:
                resp = client.messages.create(
                    model=m,
                    max_tokens=1024,
                    temperature=0.2,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}]
                )
                raw_response = resp.content[0].text
                break
            except Exception as e:
                print(f"[judge_agent warn] Anthropic model {m} failed: {e}. Trying fallback...")

    # Fallback 1: Groq API
    if not raw_response and groq_key:
        import openai
        g_client = openai.OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        for m in models_to_try:
            try:
                resp = g_client.chat.completions.create(
                    model=m,
                    temperature=0.2,
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ]
                )
                raw_response = resp.choices[0].message.content
                break
            except Exception as e:
                print(f"[judge_agent warn] Groq model {m} failed: {e}")

    # Fallback 2: OpenAI API
    if not raw_response and openai_key:
        import openai
        o_client = openai.OpenAI(api_key=openai_key)
        resp = o_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ]
        )
        raw_response = resp.choices[0].message.content

    if not raw_response:
        raise RuntimeError("No LLM API key available (ANTHROPIC_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY required).")

    res_dict = _clean_and_parse_json(raw_response)

    # Deterministically enforce Python-derived claim_is_supported in fact_check_assessment
    if "fact_check_assessment" in res_dict and isinstance(res_dict["fact_check_assessment"], dict):
        if not fc:
            res_dict["fact_check_assessment"]["performed"] = False
            res_dict["fact_check_assessment"]["claim_is_supported"] = None
        else:
            res_dict["fact_check_assessment"]["performed"] = True
            res_dict["fact_check_assessment"]["claim_is_supported"] = claim_is_supported

    return res_dict


def run_dry_run():
    """Runs pipeline.py fresh for test cases and feeds each into judge()."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pipeline

    test_cases = [
        {
            "name": "a. Earth's-age example (override wrong 'False' veracity label)",
            "text": "The Earth is approximately 4.5 billion years old.",
            "evidence": None,
        },
        {
            "name": "b. appeal_to_emotion example",
            "text": "How can you support this policy when it will break the hearts of thousands of families?",
            "evidence": None,
        },
        {
            "name": "c. fallacy_of_extension (strawman) example (disagreement with reliable=true)",
            "text": "So you're saying we should let anyone in with zero rules at all?",
            "evidence": None,
        },
        {
            "name": "d. 'Nice weather today' example (unreliable/near-zero signals)",
            "text": "Nice weather today.",
            "evidence": None,
        },
        {
            "name": "e. vaccines/autism example with evidence (NLI REFUTES -> claim_is_supported=false)",
            "text": "Vaccines cause autism.",
            "evidence": "Multiple large-scale studies have found no link between vaccines and autism.",
        },
        {
            "name": "f. Great Wall of China example with evidence (NLI SUPPORTS -> claim_is_supported=true)",
            "text": "The Great Wall of China is over 13,000 miles long.",
            "evidence": "Historical surveys measure the total length of the Great Wall of China at approximately 13,171 miles including all branches.",
        },
    ]

    print("\n============================================================")
    print(" JUDGE AGENT DRY RUN (FRESH PIPELINE OUTPUTS -> JUDGE AGENT)")
    print("============================================================")

    for tc in test_cases:
        print(f"\n============================================================")
        print(f" TEST CASE: {tc['name']}")
        print(f"============================================================")
        
        pipe_out = pipeline.analyze_text(tc["text"], tc["evidence"])
        print("\n--- FRESH PIPELINE OUTPUT ---")
        print(json.dumps(pipe_out, indent=2))

        verdict = judge(pipe_out)
        print("\n--- RAW JUDGE AGENT VERDICT JSON ---")
        print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run end-to-end dry run test suite")
    args = parser.parse_args()

    if args.dry_run or len(sys.argv) == 1:
        run_dry_run()
