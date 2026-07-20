"""
tests/conftest.py — Configuration and shared fixtures for the test suite.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from backend.db.database import Base, engine, SessionLocal
from backend.main import app

# Set event loop policy for Windows if needed
try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except AttributeError:
    pass


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    res = policy.new_event_loop()
    asyncio.set_event_loop(res)
    yield res
    res.close()


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test case."""
    # Create all tables in sqlite in-memory or file for test
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        # Drop all tables after test
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def api_client():
    """Test client for FastAPI REST endpoints."""
    return TestClient(app)
