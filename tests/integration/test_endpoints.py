"""
tests/integration/test_endpoints.py — Smoke tests for FastAPI backend routes.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.main import app
from backend.db.database import Base, engine, SessionLocal


@pytest.fixture(autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_claims_endpoint():
    client = TestClient(app)
    # Standalone claims extraction
    response = client.post("/claims", json={
        "segment_text": "The inflation rate in Chicago increased by 10% in 2019.",
        "speaker": "Alice"
    })
    assert response.status_code == 200
    data = response.json()
    assert "claims" in data
    # In mock mode, should return default claims
    assert len(data["claims"]) > 0


def test_verify_endpoint():
    client = TestClient(app)
    # Standalone claim verification
    response = client.post("/verify", json={
        "claim": "The rate of shingles is 1.2 to 3.4 per 1,000.",
        "evidence": [
            {
                "chunk_id": "c1",
                "text": "Incidence of shingles is 1.2 to 3.4 per 1,000.",
                "source_url": "http://who.int",
                "title": "WHO Shingles",
                "trust_tier": 1,
                "domain_topic": "health"
            }
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert "verdict" in data
    assert "confidence" in data
    assert "cited_chunks" in data


def test_transcribe_and_dashboard_flow():
    client = TestClient(app)

    # Patch retrieval dependencies so it runs offline/fast
    with patch("agents.retrieval_agent.SentenceTransformer"), \
         patch("sentence_transformers.CrossEncoder"), \
         patch("faiss.read_index"), \
         patch("sqlite3.connect") as mock_sqlite:
        
        mock_conn = MagicMock()
        mock_sqlite.return_value = mock_conn
        mock_cur = mock_conn.cursor.return_value
        mock_cur.fetchone.return_value = ("chunk_1", "http://test.com", "Test Title", 1, "test")

        # 1. POST /transcribe
        response = client.post("/transcribe", json={
            "segment_text": "Yesterday, the governor claimed that inflation rate in Chicago increased by 10% in 2019.",
            "speaker": "Alice",
            "session_id": "session_abc"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session_abc"
        assert len(data["claims_extracted"]) > 0

        # 2. GET /dashboard
        dashboard_resp = client.get("/dashboard?session_id=session_abc")
        assert dashboard_resp.status_code == 200
        dash_data = dashboard_resp.json()
        assert dash_data["claim_count"] > 0
        assert "Alice" in dash_data["speaker_metrics"]
        assert dash_data["speaker_metrics"]["Alice"]["claims"] > 0


def test_websocket_live():
    client = TestClient(app)
    
    # Connect to WS /live
    with client.websocket_connect("/live?session_id=ws_session") as websocket:
        # Since it runs with in-memory queue, sending a ping/keepalive or getting data works
        # If there's nothing in queue, it should block or return ping keepalive on timeout.
        # But we can push something to the queue from transcribe, and see it forwarded.
        with patch("agents.retrieval_agent.SentenceTransformer"), \
             patch("sentence_transformers.CrossEncoder"), \
             patch("faiss.read_index"), \
             patch("sqlite3.connect") as mock_sqlite:
            
            mock_conn = MagicMock()
            mock_sqlite.return_value = mock_conn
            mock_cur = mock_conn.cursor.return_value
            mock_cur.fetchone.return_value = ("chunk_1", "http://test.com", "Test Title", 1, "test")

            # Post a transcribe request to trigger a push to WS
            transcribe_resp = client.post("/transcribe", json={
                "segment_text": "WHO air pollution effects",
                "speaker": "Bob",
                "session_id": "ws_session"
            })
            assert transcribe_resp.status_code == 200
            
            # Read from WS
            message = websocket.receive_text()
            data = json.loads(message)
            assert data["session_id"] == "ws_session"
            assert data["speaker"] == "Bob"
            assert len(data["claim_results"]) > 0
            assert "summary" in data
import json
