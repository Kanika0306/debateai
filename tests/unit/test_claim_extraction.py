"""
tests/unit/test_claim_extraction.py — Unit tests for ClaimExtractionAgent.
"""
import pytest
from unittest.mock import AsyncMock, patch
from agents.claim_extraction_agent import ClaimExtractionAgent
from agents.schemas import ClaimExtractionInput, ClaimExtractionOutput


@pytest.mark.asyncio
async def test_claim_extraction_happy_path():
    agent = ClaimExtractionAgent()
    input_data = ClaimExtractionInput(
        segment_text="The inflation rate in Chicago increased by 10% in 2019.",
        speaker="Alice"
    )

    # Mock call_llm to return standard JSON
    with patch.object(
        agent,
        "call_llm",
        new_callable=AsyncMock,
        return_value='{"claims": ["The inflation rate in Chicago increased by 10% in 2019."]}'
    ) as mock_call:
        output = await agent.run(input_data)
        assert isinstance(output, ClaimExtractionOutput)
        assert len(output.claims) == 1
        assert output.claims[0] == "The inflation rate in Chicago increased by 10% in 2019."
        assert output.error is None
        mock_call.assert_called_once()


@pytest.mark.asyncio
async def test_claim_extraction_empty_input():
    agent = ClaimExtractionAgent()
    input_data = ClaimExtractionInput(
        segment_text="   ",
        speaker="Bob"
    )

    with patch.object(
        agent,
        "call_llm",
        new_callable=AsyncMock,
        return_value='{"claims": []}'
    ):
        output = await agent.run(input_data)
        assert isinstance(output, ClaimExtractionOutput)
        assert len(output.claims) == 0
        assert output.error is None


@pytest.mark.asyncio
async def test_claim_extraction_malformed_json():
    agent = ClaimExtractionAgent()
    input_data = ClaimExtractionInput(
        segment_text="Something about healthcare.",
        speaker="Alice"
    )

    # Mock call_llm to return invalid JSON
    with patch.object(
        agent,
        "call_llm",
        new_callable=AsyncMock,
        return_value='{invalid-json'
    ):
        output = await agent.run(input_data)
        assert isinstance(output, ClaimExtractionOutput)
        assert len(output.claims) == 0
        assert output.error is not None
        assert "Parsing error" in output.error


@pytest.mark.asyncio
async def test_claim_extraction_timeout_fallback():
    agent = ClaimExtractionAgent()
    input_data = ClaimExtractionInput(
        segment_text="Test timeout fallback",
        speaker="Bob"
    )

    # Mock run to simulate timeout/crash
    async def mock_run(inp):
        import asyncio
        await asyncio.sleep(2.0)
        return ClaimExtractionOutput(claims=["should not happen"])

    with patch.object(agent, "run", mock_run):
        output = await agent.run_with_timeout(input_data, timeout=0.1)
        assert isinstance(output, ClaimExtractionOutput)
        assert len(output.claims) == 0
        assert output.error == "Timeout occurred"
