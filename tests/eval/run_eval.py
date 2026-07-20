"""
tests/eval/run_eval.py — Evaluation runner for FactVerificationAgent.

Measures classification accuracy and outputs a confusion matrix.
Requires real API calls (approximately 30 calls) when running with OpenAI.
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agents.fact_verification_agent import FactVerificationAgent
from agents.schemas import FactVerificationInput, ChunkMetadata

logging.basicConfig(level=logging.WARNING)


async def run_evaluation():
    eval_file = Path("data/processed/eval_set.jsonl")
    if not eval_file.exists():
        print(f"Error: Eval file not found at {eval_file}")
        return

    # Count lines/calls
    with open(eval_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    num_calls = len(lines)
    api_key = os.environ.get("OPENAI_API_KEY")

    print("============================================================")
    print("   Debate-AI Fact Verification Evaluation Runner")
    print("============================================================")
    print(f"Loaded {num_calls} evaluation claims.")
    if api_key:
        print(f"[REAL LLM MODE] Will make ~{num_calls} real OpenAI API calls (gpt-4o-mini).")
    else:
        print(f"[OFFLINE/MOCK MODE] OPENAI_API_KEY is not set. Running with mock response generator.")
    print("============================================================\n")

    agent = FactVerificationAgent()

    # Confusion matrix tracker: gold -> predicted -> count
    classes = ["True", "False", "Misleading", "Unverified"]
    confusion_matrix = {g: {p: 0 for p in classes} for g in classes}

    correct = 0
    total = 0

    print(f"{'Claim':<50} | {'Gold':<10} | {'Predicted':<10} | {'Match':<5}")
    print("-" * 83)

    for entry in lines:
        claim = entry["claim"]
        gold_label = entry["label"]
        evidence_text = entry["evidence"]

        # Prepare input
        evidence_chunk = ChunkMetadata(
            chunk_id="gold_chunk_01",
            text=evidence_text,
            source_url="http://eval.source",
            title="Gold Evidence",
            trust_tier=1,
            domain_topic="eval"
        )
        
        inp = FactVerificationInput(claim=claim, evidence=[evidence_chunk])
        
        # Run agent
        try:
            output = await agent.run(inp)
            predicted_label = output.verdict
        except Exception as e:
            predicted_label = "Unverified"
            print(f"Error running agent on claim: {claim[:20]}... - {e}")

        # Update stats
        if predicted_label not in classes:
            # Handle potential casing or unknown values gracefully
            predicted_label = "Unverified"

        confusion_matrix[gold_label][predicted_label] += 1
        is_match = gold_label == predicted_label
        if is_match:
            correct += 1
        total += 1

        print(f"{claim[:50]:<50} | {gold_label:<10} | {predicted_label:<10} | {str(is_match):<5}")

    accuracy = correct / total if total > 0 else 0.0

    print("\n============================================================")
    print("   Evaluation Results Summary")
    print("============================================================")
    print(f"Total Evaluated : {total}")
    print(f"Correct Answers : {correct}")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print("\nConfusion Matrix (Rows = Gold, Cols = Predicted):")
    print(f"{'':<12} | {'True':<8} | {'False':<8} | {'Misleading':<10} | {'Unverified':<10}")
    print("-" * 65)
    for g in classes:
        print(
            f"{g:<12} | "
            f"{confusion_matrix[g]['True']:<8} | "
            f"{confusion_matrix[g]['False']:<8} | "
            f"{confusion_matrix[g]['Misleading']:<10} | "
            f"{confusion_matrix[g]['Unverified']:<10}"
        )
    print("============================================================")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
