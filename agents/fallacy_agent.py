import json
import logging
from pydantic import BaseModel
from agents.base_agent import BaseAgent
from agents.schemas import FallacyInput, FallacyOutput

log = logging.getLogger(__name__)

class FallacyAgent(BaseAgent):
    """
    Classifies a text segment for logical fallacies using our normalized 11-class taxonomy.
    """

    def get_fallback_output(self, input: FallacyInput, error_msg: str = "Timeout occurred") -> FallacyOutput:
        return FallacyOutput(
            text=input.text,
            fallacy_type="no fallacy",
            confidence=0.0,
            error=error_msg
        )

    async def run(self, input: FallacyInput) -> FallacyOutput:
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
