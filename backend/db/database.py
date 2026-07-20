"""
backend/db/database.py — SQLAlchemy engine and session factory.

Uses SQLite by default (no external dependencies for dev/test).
Set DATABASE_URL=postgresql://... for production.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from backend.config import settings


# SQLite needs check_same_thread=False for FastAPI's async usage
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: yields a DB session, closes on completion."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Call once at startup."""
    Base.metadata.create_all(bind=engine)
