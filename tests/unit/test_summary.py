"""
tests/unit/test_summary.py — Unit tests for SummaryAgent.
"""
import pytest
from agents.summary_agent import SummaryAgent
from agents.schemas import SummaryInput, JudgeOutput


@pytest.mark.asyncio
async def test_summary_accumulation():
    agent = SummaryAgent()

    batch_1 = SummaryInput(new_verdicts=[
        JudgeOutput(
            claim="Claim 1",
            speaker="Alice",
            verdict="True",
            confidence=0.9,
            fallacy="no fallacy"
        ),
        JudgeOutput(
            claim="Claim 2",
            speaker="Bob",
            verdict="False",
            confidence=0.8,
            fallacy="circular reasoning"
        )
    ])

    out1 = await agent.run(batch_1)
    assert out1.claim_count == 2
    assert out1.verdict_breakdown["True"] == 1
    assert out1.verdict_breakdown["False"] == 1
    assert out1.speaker_metrics["Alice"]["claims"] == 1
    assert out1.speaker_metrics["Alice"]["false_claims"] == 0
    assert out1.speaker_metrics["Alice"]["fallacies"] == 0
    assert out1.speaker_metrics["Bob"]["claims"] == 1
    assert out1.speaker_metrics["Bob"]["false_claims"] == 1
    assert out1.speaker_metrics["Bob"]["fallacies"] == 1
    assert out1.fallacy_counts["circular reasoning"] == 1

    batch_2 = SummaryInput(new_verdicts=[
        JudgeOutput(
            claim="Claim 3",
            speaker="Alice",
            verdict="False",
            confidence=0.95,
            fallacy="circular reasoning"
        )
    ])

    out2 = await agent.run(batch_2)
    assert out2.claim_count == 3
    assert out2.verdict_breakdown["False"] == 2
    assert out2.speaker_metrics["Alice"]["claims"] == 2
    assert out2.speaker_metrics["Alice"]["false_claims"] == 1
    assert out2.speaker_metrics["Alice"]["fallacies"] == 1
    assert out2.fallacy_counts["circular reasoning"] == 2
