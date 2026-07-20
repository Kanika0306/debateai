"""
backend/api/deps.py — FastAPI dependency injection.

Provides DB sessions, optional Redis connection, and a shared
Orchestrator instance (lazily initialized to avoid loading ML
models at import time).
"""
import asyncio
import logging
import os
from typing import Optional

from backend.db.database import get_db

log = logging.getLogger(__name__)

# ── Shared orchestrator (lazily initialized) ──
_orchestrator = None
_orch_lock = asyncio.Lock()


async def get_orchestrator():
    """
    Lazily initializes the Orchestrator.
    Heavy (loads embedding models), so only created once.
    """
    global _orchestrator
    if _orchestrator is None:
        async with _orch_lock:
            if _orchestrator is None:
                from agents.orchestrator import Orchestrator
                _orchestrator = Orchestrator(session_id="api", use_redis=True)
    return _orchestrator


# ── Redis connection (optional) ──
_redis_client = None


def get_redis():
    """Returns a Redis client or None if unavailable."""
    global _redis_client
    if _redis_client is None:
        try:
            import redis
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            _redis_client = redis.Redis.from_url(redis_url)
            _redis_client.ping()
        except Exception:
            _redis_client = None
    return _redis_client


# ── In-memory message queue (fallback for WS when Redis is unavailable) ──
live_queues: dict[str, asyncio.Queue] = {}


def get_live_queue(session_id: str) -> asyncio.Queue:
    """Get or create an in-memory queue for a session's live updates."""
    if session_id not in live_queues:
        live_queues[session_id] = asyncio.Queue()
    return live_queues[session_id]
