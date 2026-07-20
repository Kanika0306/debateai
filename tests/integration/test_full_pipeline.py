"""
tests/integration/test_full_pipeline.py — Integration tests for Orchestrator.

Tests full transcript segment pipeline execution and timeout fallbacks.
"""
import pytest
import time
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agents.orchestrator import Orchestrator
from agents.schemas import TranscriptSegment, OrchestratorOutput, FactVerificationOutput, FactVerificationInput


@pytest.mark.asyncio
async def test_full_pipeline_integration():
    # Patch retrieval agent initialization to run offline fast
    with patch("agents.retrieval_agent.SentenceTransformer"), \
         patch("sentence_transformers.CrossEncoder"), \
         patch("faiss.read_index") as mock_faiss, \
         patch("sqlite3.connect") as mock_sqlite:
        
        mock_index = MagicMock()
        mock_index.ntotal = 100
        mock_faiss.return_value = mock_index

        mock_conn = MagicMock()
        mock_sqlite.return_value = mock_conn
        mock_cur = mock_conn.cursor.return_value
        mock_cur.fetchone.return_value = ("chunk_1", "http://test.com", "Test Title", 1, "test")

        orch = Orchestrator(session_id="integration_test", use_redis=False)
        segment = TranscriptSegment(
            segment_text=(
                "Yesterday, the governor claimed that inflation rate in Chicago increased by 10% in 2019, which is not true. "
                "Also, we all know that nuclear power has caused millions of deaths, and that's why we need solar energy."
            ),
            speaker="Alice",
            session_id="integration_test"
        )

        start_time = time.time()
        output = await orch.process_segment(segment)
        end_time = time.time()
        
        latency = end_time - start_time
        print(f"\nIntegration test latency: {latency:.4f}s")
        
        assert isinstance(output, OrchestratorOutput)
        assert output.session_id == "integration_test"
        assert output.speaker == "Alice"
        assert len(output.claims_extracted) == 2
        assert len(output.claim_results) == 2
        
        # Verify the aggregate summary metrics
        assert output.summary.claim_count == 2
        assert "Alice" in output.summary.speaker_metrics
        
        # Ensure latency is within reasonable limits
        assert latency < 15.0


@pytest.mark.asyncio
async def test_orchestrator_timeout_fallback():
    with patch("agents.retrieval_agent.SentenceTransformer"), \
         patch("sentence_transformers.CrossEncoder"), \
         patch("faiss.read_index") as mock_faiss, \
         patch("sqlite3.connect") as mock_sqlite:

        mock_index = MagicMock()
        mock_index.ntotal = 100
        mock_faiss.return_value = mock_index

        mock_conn = MagicMock()
        mock_sqlite.return_value = mock_conn
        mock_cur = mock_conn.cursor.return_value
        mock_cur.fetchone.return_value = ("chunk_1", "http://test.com", "Test Title", 1, "test")

        orch = Orchestrator(session_id="timeout_test", use_redis=False)
        segment = TranscriptSegment(
            segment_text="Shingles cases range from 1.2 to 3.4 per 1,000.",
            speaker="Bob",
            session_id="timeout_test"
        )

        # Mock claim extraction to return exactly 1 claim
        with patch.object(orch.claim_agent, "run_with_timeout", AsyncMock(return_value=MagicMock(claims=["Shingles cases range from 1.2 to 3.4 per 1,000."]))):
            # Mock fact_verification_agent run_with_timeout to return a timeout fallback output
            fallback_verif = FactVerificationOutput(
                claim="Shingles cases range from 1.2 to 3.4 per 1,000.",
                verdict="Unverified",
                confidence=0.0,
                cited_chunks=[],
                error="Timeout occurred"
            )
            
            with patch.object(orch._fact_agent, "run_with_timeout", AsyncMock(return_value=fallback_verif)):
                output = await orch.process_segment(segment)
                
                assert isinstance(output, OrchestratorOutput)
                assert len(output.claim_results) == 1
                result = output.claim_results[0]
                
                # Check that fallback verdict was returned and didn't crash
                assert result.verification_output.verdict == "Unverified"
                assert result.verification_output.error == "Timeout occurred"
                assert result.judge_output.verdict == "Unverified"
