import json
import logging
from pydantic import BaseModel
from agents.base_agent import BaseAgent
from agents.schemas import ClaimExtractionInput, ClaimExtractionOutput

log = logging.getLogger(__name__)

class ClaimExtractionAgent(BaseAgent):
    """
    Extracts checkable factual claims from a segment of transcript.
    Uses few-shot examples from fact-verification parquets.
    """

    def get_fallback_output(self, input: ClaimExtractionInput, error_msg: str = "Timeout occurred") -> ClaimExtractionOutput:
        return ClaimExtractionOutput(claims=[], error=error_msg)

    async def run(self, input: ClaimExtractionInput) -> ClaimExtractionOutput:
        system_prompt = (
            "You are an expert fact-checker assistant. Your job is to extract checkable factual claims "
            "from segments of debate transcript. A checkable claim is a statement about a historical, "
            "scientific, economic, or demographic fact that can be verified with evidence. "
            "Ignore opinions, rhetoric, predictions, or purely subjective claims. "
            "Return the output strictly in JSON format as a list of claims under the key 'claims'."
        )

        user_prompt = f"""Extract checkable factual claims from the following segment of speech by {input.speaker}:
---
"{input.segment_text}"
---

Follow this JSON response format exactly:
{{
  "claims": [
    "extracted checkable claim 1",
    "extracted checkable claim 2"
  ]
}}

Few-shot Examples:
Example 1: "We need to fix our healthcare because the number of new cases of shingles per year extends from 1.2 to 3.4 per 1,000, which is unacceptable."
Response: {{
  "claims": ["The number of new cases of shingles per year extends from 1.2 to 3.4 per 1,000."]
}}

Example 2: "My opponent claims she is a movie star, but Gabrielle Union was in a movie, not her!"
Response: {{
  "claims": ["Gabrielle Union was in a movie."]
}}

Example 3: "Polls show that 90 percent of Americans support universal background checks for gun purchases, yet Congress does nothing."
Response: {{
  "claims": ["90 percent of Americans support universal background checks for gun purchases."]
}}

Example 4: "I heard Bernie Sanders's plan is to raise your taxes to 90 percent, which would destroy the economy."
Response: {{
  "claims": ["Bernie Sanders's plan is to raise your taxes to 90 percent."]
}}

Example 5: "My opponent said crime is down, but last year was one of the deadliest years ever for law enforcement officers."
Response: {{
  "claims": ["Last year was one of the deadliest years ever for law enforcement officers."]
}}
"""

        raw_response = await self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format={"type": "json_object"}
        )
        try:
            data = json.loads(raw_response)
            claims = data.get("claims", [])
            return ClaimExtractionOutput(claims=claims)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            log.error("Failed to parse claim extraction LLM response: %s", e)
            return self.get_fallback_output(input, f"Parsing error: {str(e)}")

# Standalone verification runner
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    async def test():
        agent = ClaimExtractionAgent()
        test_input = ClaimExtractionInput(
            segment_text="Yesterday, the governor claimed that inflation rate in Chicago increased by 10% in 2019, which is not true. Also, we all know that nuclear power has caused millions of deaths, and that's why we need solar energy.",
            speaker="Alice"
        )
        print("Running standalone ClaimExtractionAgent test...")
        output = await agent.run(test_input)
        print("Output:", output.model_dump_json(indent=2))

    asyncio.run(test())
