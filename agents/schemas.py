"""
agents/schemas.py — Unified data contract for all agent boundaries.

Every Pydantic model that crosses an agent boundary is defined here so
the full data contract is visible in one place.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from enum import Enum


# ==============================================================================
# Shared Enums
# ==============================================================================
class Verdict(str, Enum):
    TRUE = "True"
    FALSE = "False"
    MISLEADING = "Misleading"
    UNVERIFIED = "Unverified"


# ==============================================================================
# Orchestrator / Pipeline Schemas
# ==============================================================================
class TranscriptSegment(BaseModel):
    """Top-level input representing one segment of debate transcript."""
    segment_text: str = Field(description="Raw text of the transcript segment.")
    speaker: str = Field(description="Name/label of the speaker.")
    session_id: str = Field(default="default", description="Debate session identifier.")


# ==============================================================================
# Claim Extraction Agent Schemas
# ==============================================================================
class ClaimExtractionInput(BaseModel):
    segment_text: str = Field(description="Segment of debate transcript to extract claims from.")
    speaker: str = Field(description="The name of the speaker who spoke this segment.")


class ClaimExtractionOutput(BaseModel):
    claims: List[str] = Field(default_factory=list, description="List of checkable factual claims extracted.")
    error: Optional[str] = Field(None, description="Error message if extraction failed.")


# ==============================================================================
# Retrieval Agent Schemas
# ==============================================================================
class ChunkMetadata(BaseModel):
    chunk_id: str
    text: str
    source_url: str
    title: str
    trust_tier: int
    domain_topic: str
    score: float = Field(0.0, description="Rerank or cosine similarity score.")


class RetrievalInput(BaseModel):
    claim: str = Field(description="The factual claim text to retrieve context for.")


class RetrievalOutput(BaseModel):
    claim: str
    chunks: List[ChunkMetadata] = Field(default_factory=list, description="Top-k matching evidence chunks.")
    error: Optional[str] = Field(None, description="Error message if retrieval failed.")


# ==============================================================================
# Fact Verification Agent Schemas
# ==============================================================================
class FactVerificationInput(BaseModel):
    claim: str
    evidence: List[ChunkMetadata] = Field(default_factory=list)


class FactVerificationOutput(BaseModel):
    claim: str
    verdict: str = Field(description="One of: True, False, Misleading, Unverified")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    cited_chunks: List[str] = Field(default_factory=list, description="List of chunk IDs supporting the verdict.")
    error: Optional[str] = Field(None, description="Error/Timeout message if failed.")

    @field_validator("verdict")
    @classmethod
    def validate_verdict(cls, v: str) -> str:
        allowed = {"True", "False", "Misleading", "Unverified"}
        if v not in allowed:
            return "Unverified"
        return v

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# ==============================================================================
# Fallacy Agent Schemas
# ==============================================================================
FALLACY_TAXONOMY = [
    "ad hominem",
    "ad populum",
    "appeal to emotion",
    "circular reasoning",
    "false causality",
    "false dilemma",
    "hasty generalization",
    "fallacy of relevance",
    "fallacy of credibility",
    "equivocation",
    "no fallacy",
]


class FallacyInput(BaseModel):
    text: str = Field(description="Text to analyze for logical fallacies.")


class FallacyOutput(BaseModel):
    text: str
    fallacy_type: str = Field(description="Detected fallacy type from proposed taxonomy, or 'no fallacy'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    error: Optional[str] = Field(None, description="Error/Timeout message if failed.")

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# ==============================================================================
# Judge Agent Schemas
# ==============================================================================
class JudgeInput(BaseModel):
    claim: str
    speaker: str = Field(default="unknown", description="Speaker who made the claim.")
    verification: FactVerificationOutput
    fallacy: FallacyOutput


class JudgeOutput(BaseModel):
    claim: str
    speaker: str = Field(default="unknown", description="Speaker who made the claim.")
    verdict: str = Field(description="Final resolved verdict (True, False, Misleading, Unverified)")
    confidence: float = Field(description="Resolved confidence score")
    fallacy: str = Field(description="Final resolved fallacy classification or 'no fallacy'")
    cited_chunks: List[str] = Field(default_factory=list)
    action_required: bool = Field(False, description="Flag indicating if live moderator action or flag is needed.")
    error: Optional[str] = Field(None, description="Error message if execution failed.")

    @field_validator("verdict")
    @classmethod
    def validate_verdict(cls, v: str) -> str:
        allowed = {"True", "False", "Misleading", "Unverified"}
        if v not in allowed:
            return "Unverified"
        return v

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# ==============================================================================
# Summary Agent Schemas
# ==============================================================================
class SummaryInput(BaseModel):
    new_verdicts: List[JudgeOutput]


class SummaryOutput(BaseModel):
    claim_count: int = Field(0, description="Total number of claims processed in the session.")
    verdict_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of True, False, Misleading, Unverified."
    )
    speaker_metrics: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Speaker name -> {'claims': int, 'fallacies': int, 'false_claims': int}"
    )
    fallacy_counts: Dict[str, int] = Field(default_factory=dict, description="Count per fallacy type.")


# ==============================================================================
# Orchestrator Schemas
# ==============================================================================
class OrchestratorInput(BaseModel):
    """Full input to the orchestration pipeline for one transcript segment."""
    segment: TranscriptSegment


class ClaimResult(BaseModel):
    """Result for a single claim after verification + fallacy + judging."""
    judge_output: JudgeOutput
    retrieval_output: RetrievalOutput
    verification_output: FactVerificationOutput
    fallacy_output: FallacyOutput


class OrchestratorOutput(BaseModel):
    """Full output of the orchestration pipeline for one segment."""
    session_id: str
    speaker: str
    claims_extracted: List[str] = Field(default_factory=list)
    claim_results: List[ClaimResult] = Field(default_factory=list)
    summary: SummaryOutput = Field(default_factory=SummaryOutput)
    error: Optional[str] = Field(None)
