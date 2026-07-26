import json
import logging
from pydantic import BaseModel
from agents.base_agent import BaseAgent
from agents.schemas import FactVerificationInput, FactVerificationOutput, ChunkMetadata

log = logging.getLogger(__name__)

# Lazy singleton instance of LocalFactVerificationAgent
_local_fact_verification_agent = None


def get_local_fact_verification_agent():
    global _local_fact_verification_agent
    if _local_fact_verification_agent is None:
        try:
            from fact_verification.inference import LocalFactVerificationAgent
            from fact_verification.config import CFG
            import os
            ckpt_dir = os.path.join(CFG.output_dir, "best")
            if os.path.exists(ckpt_dir):
                log.info("Initializing LocalFactVerificationAgent with fine-tuned DeBERTa model from %s", ckpt_dir)
                _local_fact_verification_agent = LocalFactVerificationAgent(checkpoint_dir=ckpt_dir, threshold=0.70)
            else:
                log.warning("No fine-tuned model found at %s. Running FactVerificationAgent in pure LLM mode.", ckpt_dir)
        except Exception as e:
            log.warning("Failed to initialize LocalFactVerificationAgent (%s). Falling back to LLM.", e)
            _local_fact_verification_agent = None
    return _local_fact_verification_agent


class FactVerificationAgent(BaseAgent):
    """
    Verifies factual claims against retrieved evidence chunks using a fine-tuned local DeBERTa-v3 model,
    falling back to LLM reasoning when local confidence is under threshold (0.70).
    Produces structured verdict (True, False, Misleading, Unverified).
    """

    def get_fallback_output(self, input: FactVerificationInput, error_msg: str = "Timeout occurred") -> FactVerificationOutput:
        return FactVerificationOutput(
            claim=input.claim,
            verdict="Unverified",
            confidence=0.0,
            cited_chunks=[],
            error=error_msg
        )

    async def run(self, input: FactVerificationInput) -> FactVerificationOutput:
        # Step 1: Try fine-tuned LocalFactVerificationAgent first
        local_agent = get_local_fact_verification_agent()
        if local_agent is not None:
            try:
                local_output = await local_agent.verify(input)
                if local_output is not None and local_output.confidence >= 0.70:
                    log.info(
                        "LocalFactVerificationAgent high confidence match (verdict=%s, conf=%.2f)",
                        local_output.verdict, local_output.confidence
                    )
                    return local_output
            except Exception as e:
                log.warning("LocalFactVerificationAgent execution error: %s. Falling back to LLM.", e)

        # Step 2: Fallback to LLM verifier prompt reasoning
        if not input.evidence:
            return FactVerificationOutput(
                claim=input.claim,
                verdict="Unverified",
                confidence=1.0,
                cited_chunks=[],
                error="No evidence provided for verification."
            )

        # Build evidence text block
        evidence_blocks = []
        for i, c in enumerate(input.evidence, 1):
            evidence_blocks.append(
                f"Source {i} [ID: {c.chunk_id}] (Title: {c.title}, URL: {c.source_url}):\n{c.text}\n"
            )
        evidence_text = "\n".join(evidence_blocks)

        system_prompt = (
            "You are a professional fact-checker for an AI debate system. Your job is to verify a given claim "
            "against the provided evidence chunks. You must output a JSON object containing:\n"
            "- 'verdict': one of 'True', 'False', 'Misleading', 'Unverified'.\n"
            "- 'confidence': float between 0.0 and 1.0 indicating how strongly the evidence supports your verdict.\n"
            "- 'cited_chunks': a list of chunk IDs (from the provided evidence sources) that directly support your verdict.\n\n"
            "Verdict Guidelines:\n"
            "- True: The claim is fully and accurately supported by the evidence.\n"
            "- False: The claim is directly contradicted by the evidence.\n"
            "- Misleading: The claim contains elements of truth but ignores key details, takes facts out of context, "
            "or exaggerates to present a false impression.\n"
            "- Unverified: The evidence contains no information to confirm or deny the claim."
        )

        user_prompt = f"""Verify this claim: "{input.claim}"

Evidence Sources:
---
{evidence_text}
---

Return your response strictly in the following JSON format:
{{
  "verdict": "True/False/Misleading/Unverified",
  "confidence": 0.95,
  "cited_chunks": ["source_chunk_id_1"]
}}

Few-shot Examples:
Example 1 (True):
Claim: "The number of new cases of shingles per year extends from 1.2 to 3.4 per 1,000."
Evidence:
Source 1 [ID: who_shingles_01]: "Incidence rates of herpes zoster (shingles) range between 1.2 and 3.4 cases per 1,000 person-years."
Response: {{
  "verdict": "True",
  "confidence": 0.98,
  "cited_chunks": ["who_shingles_01"]
}}

Example 2 (False):
Claim: "Schuyler VanValkenburg cosponsored a bill that would have allowed abortion until the moment of birth."
Evidence:
Source 1 [ID: liar_abortion_03]: "The bill in question actually reduced the number of certifying doctors required for third-trimester abortions from three to one. It did not create a right to unconditional abortion up to birth."
Response: {{
  "verdict": "False",
  "confidence": 0.95,
  "cited_chunks": ["liar_abortion_03"]
}}

Example 3 (Misleading):
Claim: "Says Barack Obama robbed Medicare of $716 billion to pay for Obamacare."
Evidence:
Source 1 [ID: liar_medicare_05]: "Obamacare did cut $716 billion in future growth projections for Medicare spending over ten years. However, these cuts targeted provider payments and insurance subsidies to improve efficiency, not current beneficiary funds."
Response: {{
  "verdict": "Misleading",
  "confidence": 0.90,
  "cited_chunks": ["liar_medicare_05"]
}}

Example 4 (Unverified):
Claim: "Eleveneleven was founded by a chef."
Evidence:
Source 1 [ID: fever_eleven_01]: "Eleveneleven is a fashion and lifestyle brand known for organic cotton apparel and hand-loomed fabrics."
Response: {{
  "verdict": "Unverified",
  "confidence": 1.0,
  "cited_chunks": []
}}
"""

        raw_response = await self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format={"type": "json_object"}
        )
        try:
            data = json.loads(raw_response)
            verdict = data.get("verdict", "Unverified")
            confidence = float(data.get("confidence", 0.0))
            cited = data.get("cited_chunks", [])
            return FactVerificationOutput(
                claim=input.claim,
                verdict=verdict,
                confidence=confidence,
                cited_chunks=cited
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            log.error("Failed to parse verification LLM response: %s", e)
            return self.get_fallback_output(input, f"Parsing error: {str(e)}")

# Standalone verification runner
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    async def test():
        agent = FactVerificationAgent()
        test_input = FactVerificationInput(
            claim="The number of new cases of shingles per year extends from 1.2 to 3.4 per 1,000.",
            evidence=[
                ChunkMetadata(
                    chunk_id="who_shingles_0001",
                    text="Incidence rates of herpes zoster (shingles) range between 1.2 and 3.4 cases per 1,000 person-years.",
                    source_url="https://who.int/shingles",
                    title="Shingles fact sheet",
                    trust_tier=1,
                    domain_topic="health"
                )
            ]
        )
        print("\nRunning standalone FactVerificationAgent test...")
        output = await agent.run(test_input)
        print("Output:", output.model_dump_json(indent=2))

    asyncio.run(test())
