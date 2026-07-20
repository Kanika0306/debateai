"""
tests/unit/test_fact_verification.py — Unit tests for FactVerificationAgent.
"""
import pytest
from unittest.mock import AsyncMock, patch
from agents.fact_verification_agent import FactVerificationAgent
from agents.schemas import FactVerificationInput, FactVerificationOutput, ChunkMetadata


@pytest.fixture
def dummy_evidence():
    return [
        ChunkMetadata(
            chunk_id="chunk_1",
            text="Incidence of shingles is 1.2 to 3.4 per 1,000.",
            source_url="http://who.int",
            title="WHO Shingles",
            trust_tier=1,
            domain_topic="health"
        )
    ]


@pytest.mark.asyncio
async def test_fact_verification_happy_path(dummy_evidence):
    agent = FactVerificationAgent()
    input_data = FactVerificationInput(
        claim="Shingles rate is 1.2 to 3.4 per 1,000.",
        evidence=dummy_evidence
    )

    with patch.object(
        agent,
        "call_llm",
        new_callable=AsyncMock,
        return_value='{"verdict": "True", "confidence": 0.98, "cited_chunks": ["chunk_1"]}'
    ) as mock_call:
        output = await agent.run(input_data)
        assert isinstance(output, FactVerificationOutput)
        assert output.verdict == "True"
        assert output.confidence == 0.98
        assert output.cited_chunks == ["chunk_1"]
        assert output.error is None
        mock_call.assert_called_once()


@pytest.mark.asyncio
async def test_fact_verification_empty_evidence():
    agent = FactVerificationAgent()
    input_data = FactVerificationInput(
        claim="Shingles rate is high.",
        evidence=[]
    )

    output = await agent.run(input_data)
    assert isinstance(output, FactVerificationOutput)
    assert output.verdict == "Unverified"
    assert output.confidence == 1.0
    assert len(output.cited_chunks) == 0
    assert "No evidence provided" in output.error


@pytest.mark.asyncio
async def test_fact_verification_malformed_json(dummy_evidence):
    agent = FactVerificationAgent()
    input_data = FactVerificationInput(
        claim="Shingles rate is 1.2 to 3.4 per 1,000.",
        evidence=dummy_evidence
    )

    with patch.object(
        agent,
        "call_llm",
        new_callable=AsyncMock,
        return_value='{invalid-json'
    ):
        output = await agent.run(input_data)
        assert isinstance(output, FactVerificationOutput)
        assert output.verdict == "Unverified"
        assert output.confidence == 0.0
        assert len(output.cited_chunks) == 0
        assert "Parsing error" in output.error
