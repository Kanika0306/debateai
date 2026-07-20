"""
tests/unit/test_retrieval.py — Unit tests for RetrievalAgent.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from agents.retrieval_agent import RetrievalAgent
from agents.schemas import RetrievalInput, RetrievalOutput, ChunkMetadata


@pytest.fixture
def mock_retrieval_agent():
    # Patch __init__ of RetrievalAgent to skip loading actual models
    with patch("agents.retrieval_agent.SentenceTransformer") as mock_st, \
         patch("sentence_transformers.CrossEncoder") as mock_ce, \
         patch("faiss.read_index") as mock_faiss:
        
        agent = RetrievalAgent()
        agent.index = MagicMock()
        agent.index.ntotal = 10
        agent.embed_model = MagicMock()
        agent.reranker = MagicMock()
        agent.chunk_texts = {"chunk_1": "WHO air pollution is bad."}
        return agent


@pytest.mark.asyncio
async def test_retrieval_happy_path(mock_retrieval_agent):
    agent = mock_retrieval_agent
    input_data = RetrievalInput(claim="WHO air pollution health effects")

    # Mock embed_model.encode to return dummy embedding
    agent.embed_model.encode.return_value = np.zeros((1, 512), dtype=np.float32)
    # Mock index.search to return top indices
    agent.index.search.return_value = (np.array([[0.95]]), np.array([[123]]))

    # Mock sqlite3 context
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value
    mock_cur.fetchone.side_effect = [
        ("chunk_1", "http://who.int", "WHO fact sheet", 1, "health"),  # First call (metadata)
        ("WHO air pollution is bad.",)                                  # Second call (text preview fallback if text is not in chunk_texts)
    ]

    with patch("sqlite3.connect", return_value=mock_conn):
        output = await agent.run(input_data)
        assert isinstance(output, RetrievalOutput)
        assert len(output.chunks) == 1
        assert output.chunks[0].chunk_id == "chunk_1"
        assert output.chunks[0].trust_tier == 1
        assert output.chunks[0].score == 0.95
        assert output.error is None


@pytest.mark.asyncio
async def test_retrieval_empty_input(mock_retrieval_agent):
    agent = mock_retrieval_agent
    input_data = RetrievalInput(claim="   ")
    
    output = await agent.run(input_data)
    assert isinstance(output, RetrievalOutput)
    assert len(output.chunks) == 0
    assert output.error is None


@pytest.mark.asyncio
async def test_retrieval_cache_hits(mock_retrieval_agent):
    agent = mock_retrieval_agent
    claim = "caching test claim"
    input_data = RetrievalInput(claim=claim)

    agent.embed_model.encode.return_value = np.zeros((1, 512), dtype=np.float32)
    agent.index.search.return_value = (np.array([[0.9]]), np.array([[1]]))
    
    mock_conn = MagicMock()
    mock_cur = mock_conn.cursor.return_value
    mock_cur.fetchone.return_value = ("chunk_cache", "http://test.com", "Test Title", 2, "test")

    with patch("sqlite3.connect", return_value=mock_conn):
        # First call: populates cache
        out1 = await agent.run(input_data)
        assert len(out1.chunks) == 1
        
        # Second call: should hit cache directly
        out2 = await agent.run(input_data)
        assert len(out2.chunks) == 1
        assert out1.chunks[0].chunk_id == out2.chunks[0].chunk_id
        
        # Verify encode was only called once
        agent.embed_model.encode.assert_called_once()
