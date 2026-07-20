"""
tests/unit/test_judge.py — Unit tests for JudgeAgent.
"""
import pytest
from agents.judge_agent import JudgeAgent
from agents.schemas import JudgeInput, FactVerificationOutput, FallacyOutput, JudgeOutput


@pytest.mark.asyncio
async def test_judge_downgrade_low_confidence():
    agent = JudgeAgent()
    input_data = JudgeInput(
        claim="Low confidence test.",
        verification=FactVerificationOutput(
            claim="Low confidence test.",
            verdict="False",
            confidence=0.4,  # Under 0.5 threshold
            cited_chunks=[]
        ),
        fallacy=FallacyOutput(
            text="Low confidence test.",
            fallacy_type="no fallacy",
            confidence=0.9
        )
    )
    
    output = await agent.run(input_data)
    assert isinstance(output, JudgeOutput)
    assert output.verdict == "Unverified"
    assert output.confidence == 0.4
    assert output.action_required is False


@pytest.mark.asyncio
async def test_judge_action_required_false_claim():
    agent = JudgeAgent()
    input_data = JudgeInput(
        claim="Some false claim.",
        verification=FactVerificationOutput(
            claim="Some false claim.",
            verdict="False",
            confidence=0.85,
            cited_chunks=["c1"]
        ),
        fallacy=FallacyOutput(
            text="Some false claim.",
            fallacy_type="no fallacy",
            confidence=0.9
        )
    )
    
    output = await agent.run(input_data)
    assert output.verdict == "False"
    assert output.action_required is True


@pytest.mark.asyncio
async def test_judge_action_required_fallacy():
    agent = JudgeAgent()
    input_data = JudgeInput(
        claim="Rhetorical attack.",
        verification=FactVerificationOutput(
            claim="Rhetorical attack.",
            verdict="True",
            confidence=0.9,
            cited_chunks=["c1"]
        ),
        fallacy=FallacyOutput(
            text="Rhetorical attack.",
            fallacy_type="ad hominem",
            confidence=0.85
        )
    )
    
    output = await agent.run(input_data)
    assert output.verdict == "True"
    assert output.fallacy == "ad hominem"
    assert output.action_required is True


@pytest.mark.asyncio
async def test_judge_propagate_errors():
    agent = JudgeAgent()
    input_data = JudgeInput(
        claim="Error propagation.",
        verification=FactVerificationOutput(
            claim="Error propagation.",
            verdict="Unverified",
            confidence=0.0,
            error="Verif Timeout"
        ),
        fallacy=FallacyOutput(
            text="Error propagation.",
            fallacy_type="no fallacy",
            confidence=0.0,
            error="Fallacy Crash"
        )
    )
    
    output = await agent.run(input_data)
    assert output.verdict == "Unverified"
    assert "verification: Verif Timeout" in output.error
    assert "fallacy: Fallacy Crash" in output.error
