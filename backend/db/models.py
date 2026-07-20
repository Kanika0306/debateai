"""
backend/db/models.py — SQLAlchemy ORM models.

Tables: sessions, transcripts, claims, verdicts, fallacies.
Fields match agent output schemas so no reshaping is needed
between the agent layer and persistence.
"""
import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from backend.db.database import Base


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(128), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    transcripts = relationship("Transcript", back_populates="session", cascade="all, delete-orphan")


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(128), ForeignKey("sessions.session_id"), nullable=False, index=True)
    speaker = Column(String(256), nullable=False)
    segment_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    session = relationship("Session", back_populates="transcripts")
    claims = relationship("Claim", back_populates="transcript", cascade="all, delete-orphan")


class Claim(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transcript_id = Column(Integer, ForeignKey("transcripts.id"), nullable=False, index=True)
    claim_text = Column(Text, nullable=False)
    speaker = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    transcript = relationship("Transcript", back_populates="claims")
    verdict = relationship("Verdict", back_populates="claim", uselist=False, cascade="all, delete-orphan")
    fallacy = relationship("Fallacy", back_populates="claim", uselist=False, cascade="all, delete-orphan")


class Verdict(Base):
    __tablename__ = "verdicts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False, unique=True, index=True)
    verdict = Column(String(32), nullable=False)  # True, False, Misleading, Unverified
    confidence = Column(Float, nullable=False)
    cited_chunks = Column(JSON, default=list)  # list[str] of chunk_ids
    action_required = Column(Boolean, default=False)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    claim = relationship("Claim", back_populates="verdict")


class Fallacy(Base):
    __tablename__ = "fallacies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False, unique=True, index=True)
    fallacy_type = Column(String(64), nullable=False)  # from normalized taxonomy
    confidence = Column(Float, nullable=False)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    claim = relationship("Claim", back_populates="fallacy")
