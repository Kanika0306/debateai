"""
backend/main.py — FastAPI application entry point.

Initializes the database, registers routes, and configures middleware.
Run with: uvicorn backend.main:app --reload
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db.database import init_db
from backend.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logging.getLogger(__name__).info("Starting debate-AI backend...")
    init_db()
    logging.getLogger(__name__).info("Database tables initialized.")
    yield
    logging.getLogger(__name__).info("Shutting down debate-AI backend.")


app = FastAPI(
    title="Debate-AI Live Fact Checker",
    description="Real-time debate fact-checking, fallacy detection, and live dashboard.",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS — allow all origins in dev (tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
