"""
backend/api/routes.py — FastAPI routes for the debate-AI backend.

Endpoints:
  POST /transcribe   — Submit a transcript segment for processing
  POST /claims       — Extract claims from text (standalone)
  POST /verify       — Verify a single claim (standalone)
  GET  /dashboard    — Session summary / dashboard data
  WS   /live         — WebSocket for live streaming results
"""
import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session as DBSession

from agents.schemas import (
    TranscriptSegment,
    ClaimExtractionInput,
    ClaimExtractionOutput,
    FactVerificationInput,
    FactVerificationOutput,
    ChunkMetadata,
    OrchestratorOutput,
    SummaryOutput,
)
from backend.db.database import get_db
from backend.db import models
from backend.api.deps import get_orchestrator, get_redis, get_live_queue

log = logging.getLogger(__name__)

router = APIRouter()


# ==============================================================================
# POST /transcribe — Full pipeline: transcript -> claims -> verify -> judge
# ==============================================================================
class TranscribeRequest(TranscriptSegment):
    """Alias for clarity at the API boundary."""
    pass


@router.post("/transcribe", response_model=dict)
async def transcribe(req: TranscribeRequest, db: DBSession = Depends(get_db)):
    """
    Process a transcript segment through the full pipeline:
    extract claims → retrieve evidence → verify → detect fallacies → judge → summarize.
    """
    orch = await get_orchestrator()

    # Ensure session exists in DB
    db_session = db.query(models.Session).filter_by(session_id=req.session_id).first()
    if not db_session:
        db_session = models.Session(session_id=req.session_id)
        db.add(db_session)
        db.commit()

    # Persist transcript
    transcript = models.Transcript(
        session_id=req.session_id,
        speaker=req.speaker,
        segment_text=req.segment_text,
    )
    db.add(transcript)
    db.commit()
    db.refresh(transcript)

    # Run pipeline
    output = await orch.process_segment(req)

    # Persist claim results
    for cr in output.claim_results:
        j = cr.judge_output
        claim_row = models.Claim(
            transcript_id=transcript.id,
            claim_text=j.claim,
            speaker=j.speaker,
        )
        db.add(claim_row)
        db.commit()
        db.refresh(claim_row)

        verdict_row = models.Verdict(
            claim_id=claim_row.id,
            verdict=j.verdict,
            confidence=j.confidence,
            cited_chunks=j.cited_chunks,
            action_required=j.action_required,
            error=j.error,
        )
        db.add(verdict_row)

        fallacy_row = models.Fallacy(
            claim_id=claim_row.id,
            fallacy_type=j.fallacy,
            confidence=cr.fallacy_output.confidence,
            error=cr.fallacy_output.error,
        )
        db.add(fallacy_row)

    db.commit()

    # Push to live queue for WS subscribers
    queue = get_live_queue(req.session_id)
    await queue.put(output.model_dump_json())

    return output.model_dump()


# ==============================================================================
# POST /claims — Extract claims from text (standalone endpoint)
# ==============================================================================
@router.post("/claims", response_model=dict)
async def extract_claims(req: ClaimExtractionInput):
    """Extract checkable factual claims from a text segment."""
    from agents.claim_extraction_agent import ClaimExtractionAgent
    agent = ClaimExtractionAgent()
    output = await agent.run_with_timeout(req, timeout=10.0)
    return output.model_dump()


# ==============================================================================
# POST /verify — Verify a single claim (standalone endpoint)
# ==============================================================================
class VerifyRequest(FactVerificationInput):
    """Alias for clarity at the API boundary."""
    pass


@router.post("/verify", response_model=dict)
async def verify_claim(req: VerifyRequest):
    """Verify a single claim against provided evidence."""
    from agents.fact_verification_agent import FactVerificationAgent
    agent = FactVerificationAgent()
    output = await agent.run_with_timeout(req, timeout=10.0)
    return output.model_dump()


# ==============================================================================
# GET /dashboard — Session summary
# ==============================================================================
@router.get("/dashboard", response_model=dict)
async def dashboard(
    session_id: str = Query(default="default"),
    db: DBSession = Depends(get_db),
):
    """
    Returns aggregated dashboard data for a session:
    claim count, verdict breakdown, speaker metrics, fallacy counts.
    """
    db_session = db.query(models.Session).filter_by(session_id=session_id).first()
    if not db_session:
        return SummaryOutput().model_dump()

    # Aggregate from DB
    claims = (
        db.query(models.Claim)
        .join(models.Transcript)
        .filter(models.Transcript.session_id == session_id)
        .all()
    )

    verdict_breakdown = {"True": 0, "False": 0, "Misleading": 0, "Unverified": 0}
    speaker_metrics = {}
    fallacy_counts = {}

    for claim in claims:
        # Verdict
        if claim.verdict:
            v = claim.verdict.verdict
            if v in verdict_breakdown:
                verdict_breakdown[v] += 1

        # Speaker metrics
        speaker = claim.speaker or "unknown"
        if speaker not in speaker_metrics:
            speaker_metrics[speaker] = {"claims": 0, "fallacies": 0, "false_claims": 0}
        speaker_metrics[speaker]["claims"] += 1
        if claim.verdict and claim.verdict.verdict == "False":
            speaker_metrics[speaker]["false_claims"] += 1
        if claim.fallacy and claim.fallacy.fallacy_type != "no fallacy":
            speaker_metrics[speaker]["fallacies"] += 1
            ft = claim.fallacy.fallacy_type
            fallacy_counts[ft] = fallacy_counts.get(ft, 0) + 1

    return SummaryOutput(
        claim_count=len(claims),
        verdict_breakdown=verdict_breakdown,
        speaker_metrics=speaker_metrics,
        fallacy_counts=fallacy_counts,
    ).model_dump()


# ==============================================================================
# WS /live — WebSocket for live streaming results
# ==============================================================================
@router.websocket("/live")
async def websocket_live(
    websocket: WebSocket,
    session_id: str = Query(default="default"),
):
    """
    WebSocket endpoint that streams live pipeline results.
    Subscribes to Redis pub/sub or in-memory queue.
    """
    await websocket.accept()
    log.info("WS /live: client connected for session %s", session_id)

    redis_client = get_redis()
    channel = f"debate:{session_id}:live"

    try:
        if redis_client:
            # Redis pub/sub path
            pubsub = redis_client.pubsub()
            pubsub.subscribe(channel)
            try:
                while True:
                    msg = pubsub.get_message(timeout=1.0)
                    if msg and msg["type"] == "message":
                        await websocket.send_text(msg["data"].decode("utf-8"))
                    # Also check for messages from client (ping/close)
                    try:
                        data = await asyncio.wait_for(
                            websocket.receive_text(), timeout=0.1
                        )
                    except asyncio.TimeoutError:
                        pass
                    except WebSocketDisconnect:
                        break
            finally:
                pubsub.unsubscribe(channel)
        else:
            # In-memory queue fallback
            queue = get_live_queue(session_id)
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                    await websocket.send_text(payload)
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    await websocket.send_json({"type": "ping"})
                except WebSocketDisconnect:
                    break

    except WebSocketDisconnect:
        log.info("WS /live: client disconnected for session %s", session_id)
    except Exception as e:
        log.error("WS /live: error: %s", e)
    finally:
        log.info("WS /live: cleanup for session %s", session_id)
