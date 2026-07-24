import json
import logging
from typing import Optional
from pathlib import Path

from agents.base_agent import BaseAgent
from agents.schemas import FallacyInput, FallacyOutput

log = logging.getLogger(__name__)

# Lazy singleton instance of LocalFallacyAgent
_local_fallacy_agent = None

def get_local_fallacy_agent():
    global _local_fallacy_agent
    if _local_fallacy_agent is None:
        try:
            from fallacy_classifier.inference import LocalFallacyAgent
            from fallacy_classifier.config import FINAL_MODEL_DIR
            if FINAL_MODEL_DIR.exists():
                log.info("Initializing LocalFallacyAgent with fine-tuned DeBERTa model from %s", FINAL_MODEL_DIR)
                _local_fallacy_agent = LocalFallacyAgent(model_dir=FINAL_MODEL_DIR, threshold=0.55)
            else:
                log.warning("No fine-tuned model found at %s. Running FallacyAgent in pure LLM mode.", FINAL_MODEL_DIR)
        except Exception as e:
            log.warning("Failed to initialize LocalFallacyAgent (%s). Falling back to LLM.", e)
            _local_fallacy_agent = None
    return _local_fallacy_agent


class FallacyAgent(BaseAgent):
    """
    Classifies a text segment for logical fallacies using our normalized 11-class taxonomy.
    Employs a local DeBERTa-v3 model with confidence-threshold fallback to LLM.
    """

    def get_fallback_output(self, input: FallacyInput, error_msg: str = "Timeout occurred") -> FallacyOutput:
        return FallacyOutput(
            text=input.text,
            fallacy_type="no fallacy",
            confidence=0.0,
            error=error_msg
        )

    async def run(self, input: FallacyInput) -> FallacyOutput:
        # Step 1: Try fine-tuned LocalFallacyAgent first
        local_agent = get_local_fallacy_agent()
        if local_agent is not None:
            try:
                local_results = await local_agent.analyze(input.text)
                if local_results:
                    # Pick highest confidence flag
                    top_flag = max(local_results, key=lambda x: x["confidence"])
                    if top_flag["confidence"] >= 0.65:
                        log.info(
                            "LocalFallacyAgent high confidence match (%s, conf=%.2f)",
                            top_flag["fallacy_type"], top_flag["confidence"]
                        )
                        return FallacyOutput(
                            text=top_flag["text"],
                            fallacy_type=top_flag["fallacy_type"],
                            confidence=top_flag["confidence"]
                        )
            except Exception as e:
                log.warning("LocalFallacyAgent execution error: %s. Falling back to LLM.", e)

        # Step 2: Fallback to LLM agent prompt reasoning
        system_prompt = (
            "You are an expert logician. Your job is to classify the provided argument "
            "into exactly one of the logical fallacies in our normalized taxonomy:\n\n"
            "Taxonomy:\n"
            "- 'ad hominem': Attacking the opponent's character or motives instead of the argument.\n"
            "- 'ad populum': Appeal to popularity (claiming something is true/good because many people believe/do it).\n"
            "- 'appeal to emotion': Manipulating emotions (fear, pity, anger) rather than logical evidence.\n"
            "- 'circular reasoning': The premise assumes the truth of the conclusion it is trying to prove.\n"
            "- 'hasty generalization': Reaching a general conclusion based on too small of a sample.\n"
            "- 'false causality': Assuming that because one event followed another, it was caused by it.\n"
            "- 'false dilemma': Presenting only two choices when other valid options exist.\n"
            "- 'fallacy of relevance': Introducing irrelevant ideas to distract (including red herrings).\n"
            "- 'fallacy of credibility': Appealing to false, unqualified, or irrelevant authorities.\n"
            "- 'equivocation': Using ambiguous language with double meanings to deceive or mislead.\n"
            "- 'no fallacy': A logically valid, sound argument.\n\n"
            "Output must be a JSON object containing:\n"
            "- 'fallacy_type': string matching one of the exact names above.\n"
            "- 'confidence': float between 0.0 and 1.0."
        )

        user_prompt = f"""Classify the following text for logical fallacies:
---
"{input.text}"
---

Follow this JSON response format exactly:
{{
  "fallacy_type": "ad populum",
  "confidence": 0.95
}}
"""

        raw_response = await self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format={"type": "json_object"}
        )
        try:
            data = json.loads(raw_response)
            ftype = data.get("fallacy_type", "no fallacy").strip().lower()
            confidence = float(data.get("confidence", 0.0))
            return FallacyOutput(
                text=input.text,
                fallacy_type=ftype,
                confidence=confidence
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            log.error("Failed to parse fallacy LLM response: %s", e)
            return self.get_fallback_output(input, f"Parsing error: {str(e)}")


# Standalone verification runner
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    async def test():
        agent = FallacyAgent()
        test_input = FallacyInput(
            text="Everyone else is doing it, so you should too."
        )
        print("\nRunning standalone FallacyAgent test...")
        output = await agent.run(test_input)
        print("Output:", output.model_dump_json(indent=2))

    asyncio.run(test())
