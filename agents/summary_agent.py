"""
agents/summary_agent.py — Session-level aggregation of claim results.

Pure Python aggregation (no LLM call): maintains running counts of
claims, verdicts by class, per-speaker metrics, and fallacy frequency.
"""
import logging
from agents.base_agent import BaseAgent
from agents.schemas import SummaryInput, SummaryOutput, JudgeOutput

log = logging.getLogger(__name__)


class SummaryAgent(BaseAgent):
    """
    Accumulates judge outputs into a running session-level summary.
    Stateful: call run() repeatedly as new batches arrive, and it
    updates internal counters. No LLM needed — this is plain aggregation.
    """

    def __init__(self):
        super().__init__()
        self._claim_count: int = 0
        self._verdict_breakdown: dict[str, int] = {
            "True": 0, "False": 0, "Misleading": 0, "Unverified": 0
        }
        self._speaker_metrics: dict[str, dict[str, int]] = {}
        self._fallacy_counts: dict[str, int] = {}

    def reset(self):
        """Reset all counters (e.g. new session)."""
        self._claim_count = 0
        self._verdict_breakdown = {
            "True": 0, "False": 0, "Misleading": 0, "Unverified": 0
        }
        self._speaker_metrics = {}
        self._fallacy_counts = {}

    def get_fallback_output(
        self, input: SummaryInput, error_msg: str = "Timeout occurred"
    ) -> SummaryOutput:
        # Return current snapshot even on error
        return SummaryOutput(
            claim_count=self._claim_count,
            verdict_breakdown=dict(self._verdict_breakdown),
            speaker_metrics={k: dict(v) for k, v in self._speaker_metrics.items()},
            fallacy_counts=dict(self._fallacy_counts),
        )

    async def run(self, input: SummaryInput) -> SummaryOutput:
        for v in input.new_verdicts:
            self._accumulate(v)

        return SummaryOutput(
            claim_count=self._claim_count,
            verdict_breakdown=dict(self._verdict_breakdown),
            speaker_metrics={k: dict(v) for k, v in self._speaker_metrics.items()},
            fallacy_counts=dict(self._fallacy_counts),
        )

    def _accumulate(self, v: JudgeOutput):
        """Update all counters with one JudgeOutput."""
        self._claim_count += 1

        # Verdict breakdown
        verdict_key = v.verdict if v.verdict in self._verdict_breakdown else "Unverified"
        self._verdict_breakdown[verdict_key] += 1

        # Speaker metrics
        speaker = v.speaker or "unknown"
        if speaker not in self._speaker_metrics:
            self._speaker_metrics[speaker] = {
                "claims": 0, "fallacies": 0, "false_claims": 0
            }
        self._speaker_metrics[speaker]["claims"] += 1
        if v.verdict == "False":
            self._speaker_metrics[speaker]["false_claims"] += 1
        if v.fallacy and v.fallacy != "no fallacy":
            self._speaker_metrics[speaker]["fallacies"] += 1

        # Fallacy counts
        if v.fallacy and v.fallacy != "no fallacy":
            self._fallacy_counts[v.fallacy] = self._fallacy_counts.get(v.fallacy, 0) + 1


# Standalone verification runner
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def test():
        agent = SummaryAgent()

        batch = SummaryInput(new_verdicts=[
            JudgeOutput(
                claim="Inflation rose 10% in 2019.",
                speaker="Alice",
                verdict="False",
                confidence=0.9,
                fallacy="hasty generalization",
                cited_chunks=["wb_inflation_01"],
                action_required=True,
            ),
            JudgeOutput(
                claim="WHO recommends 150 min of exercise per week.",
                speaker="Bob",
                verdict="True",
                confidence=0.95,
                fallacy="no fallacy",
                cited_chunks=["who_exercise_01"],
                action_required=False,
            ),
            JudgeOutput(
                claim="Solar energy is cheaper than coal.",
                speaker="Alice",
                verdict="Misleading",
                confidence=0.7,
                fallacy="no fallacy",
                cited_chunks=["nasa_energy_02"],
                action_required=False,
            ),
        ])

        print("Running standalone SummaryAgent test...")
        output = await agent.run(batch)
        print("Output:", output.model_dump_json(indent=2))

        # Second batch to test accumulation
        batch2 = SummaryInput(new_verdicts=[
            JudgeOutput(
                claim="5G causes COVID.",
                speaker="Bob",
                verdict="False",
                confidence=0.99,
                fallacy="false causality",
                cited_chunks=["who_5g_01"],
                action_required=True,
            ),
        ])
        output2 = await agent.run(batch2)
        print("\nAfter second batch:")
        print("Output:", output2.model_dump_json(indent=2))

    asyncio.run(test())
