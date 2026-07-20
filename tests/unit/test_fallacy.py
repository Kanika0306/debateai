"""
tests/unit/test_fallacy.py — Unit tests for FallacyAgent.
"""
import pytest
from unittest.mock import AsyncMock, patch
from agents.fallacy_agent import FallacyAgent
from agents.schemas import FallacyInput, FallacyOutput


@pytest.mark.asyncio
async def test_fallacy_happy_path():
    agent = FallacyAgent()
    input_data = FallacyInput(text="Everyone else is doing it, so you should too.")

    with patch.object(
        agent,
        "call_llm",
        new_callable=AsyncMock,
        return_value='{"fallacy_type": "ad populum", "confidence": 0.95}'
    ) as mock_call:
        output = await agent.run(input_data)
        assert isinstance(output, FallacyOutput)
        assert output.fallacy_type == "ad populum"
        assert output.confidence == 0.95
        assert output.error is None
        mock_call.assert_called_once()


@pytest.mark.asyncio
async def test_fallacy_malformed_json():
    agent = FallacyAgent()
    input_data = FallacyInput(text="Some text.")

    with patch.object(
        agent,
        "call_llm",
        new_callable=AsyncMock,
        return_value='{bad-json'
    ):
        output = await agent.run(input_data)
        assert isinstance(output, FallacyOutput)
        assert output.fallacy_type == "no fallacy"
        assert output.confidence == 0.0
        assert "Parsing error" in output.error
