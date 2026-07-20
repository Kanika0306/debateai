"""
backend/config.py — Application settings loaded from environment / .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


class Settings:
    """Simple settings container; reads from environment variables."""

    # Database — defaults to SQLite for local dev, set DATABASE_URL for Postgres
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{_ROOT / 'debate_ai.db'}"
    )

    # Redis — optional; in-memory fallback used when unavailable
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Agent defaults
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "8"))
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "10.0"))


settings = Settings()
