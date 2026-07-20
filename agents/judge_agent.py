"""
agents/judge_agent.py — Resolves fact-verification and fallacy results
into a final combined verdict per claim.

Pure Python logic (no LLM call needed): merges verification + fallacy
outputs, handles low-confidence verdicts, and flags action-required items.
"""
import logging
from agents.base_agent import BaseAgent
from agents.schemas import JudgeInput, JudgeOutput

log = logging.getLogger(__name__)


class JudgeAgent(BaseAgent):
    """
    Takes FactVerificationOutput + FallacyOutput for a single claim,
    resolves conflicts, and produces the final JudgeOutput.
    """

    def get_fallback_output(
        self, input: JudgeInput, error_msg: str = "Timeout occurred"
    ) -> JudgeOutput:
        return JudgeOutput(
            claim=input.claim,
            speaker=input.speaker,
            verdict="Unverified",
            confidence=0.0,
            fallacy="no fallacy",
            cited_chunks=[],
            action_required=False,
            error=error_msg,
        )

    async def run(self, input: JudgeInput) -> JudgeOutput:
        ver = input.verification
        fal = input.fallacy

        # ── Start from the raw verification result ──
        verdict = ver.verdict
        confidence = ver.confidence
        cited_chunks = list(ver.cited_chunks)
        fallacy_type = fal.fallacy_type if fal.fallacy_type else "no fallacy"
        action_required = False

        # ── Rule 1: Low-confidence verification → downgrade to Unverified ──
        if confidence < 0.5 and verdict != "Unverified":
            log.info(
                "Judge: downgrading verdict %r to Unverified (confidence=%.2f < 0.5)",
                verdict, confidence,
            )
            verdict = "Unverified"

        # ── Rule 2: Propagate errors from upstream agents ──
        errors = []
        if ver.error:
            errors.append(f"verification: {ver.error}")
        if fal.error:
            errors.append(f"fallacy: {fal.error}")
        # If both upstream agents errored, the result is unusable
        if ver.error and fal.error:
            return JudgeOutput(
                claim=input.claim,
                speaker=input.speaker,
                verdict="Unverified",
                confidence=0.0,
                fallacy="no fallacy",
                cited_chunks=[],
                action_required=False,
                error="; ".join(errors),
            )

        # ── Rule 3: Flag action_required for concerning patterns ──
        # a) Claim is False with high confidence → moderator should know
        if verdict == "False" and confidence >= 0.8:
            action_required = True

        # b) Claim is Misleading with high confidence
        if verdict == "Misleading" and confidence >= 0.85:
            action_required = True

        # c) A strong fallacy detected (even if claim is True, the rhetoric
        #    is problematic and worth flagging)
        if fallacy_type != "no fallacy" and fal.confidence >= 0.8:
            action_required = True

        # d) Both False/Misleading verdict AND a logical fallacy → highest urgency
        if verdict in ("False", "Misleading") and fallacy_type != "no fallacy":
            action_required = True

        error_msg = "; ".join(errors) if errors else None

        return JudgeOutput(
            claim=input.claim,
            speaker=input.speaker,
            verdict=verdict,
            confidence=confidence,
            fallacy=fallacy_type,
            cited_chunks=cited_chunks,
            action_required=action_required,
            error=error_msg,
        )


# Standalone verification runner
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    from agents.schemas import FactVerificationOutput, FallacyOutput

    async def test():
        agent = JudgeAgent()

        # Test case: False claim + ad hominem fallacy
        test_input = JudgeInput(
            claim="Nuclear power has caused millions of deaths.",
            speaker="Bob",
            verification=FactVerificationOutput(
                claim="Nuclear power has caused millions of deaths.",
                verdict="False",
                confidence=0.92,
                cited_chunks=["who_nuclear_01", "nasa_energy_03"],
            ),
            fallacy=FallacyOutput(
                text="Nuclear power has caused millions of deaths.",
                fallacy_type="hasty generalization",
                confidence=0.85,
            ),
        )
        print("Running standalone JudgeAgent test...")
        output = await agent.run(test_input)
        print("Output:", output.model_dump_json(indent=2))

    asyncio.run(test())
