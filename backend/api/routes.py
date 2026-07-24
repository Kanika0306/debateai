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
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, Query, File, UploadFile, Form
from sqlalchemy.orm import Session as DBSession

ROOT = Path(__file__).resolve().parent.parent.parent

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


def auto_enroll_samples():
    """Enroll baseline speakers from the voxceleb_sample folder."""
    from backend.services import audio_service
    sample_dir = os.path.join(ROOT, "data", "raw", "diarization", "voxceleb_sample")
    
    if audio_service._enrolled_speakers:
        return
        
    enrollments = [
        ("Speaker_A", "sample_001_ident.wav"),
        ("Speaker_B", "sample_002_verif.wav"),
    ]
    for name, filename in enrollments:
        path = os.path.join(sample_dir, filename)
        if os.path.exists(path):
            audio_service.enroll_speaker(name, path)


# ==============================================================================
# POST /audio/enroll — Enroll a speaker profile with an audio file
# ==============================================================================
@router.post("/audio/enroll")
async def enroll_speaker(speaker_name: str, file: UploadFile = File(...)):
    """Enroll a speaker profile by uploading a reference audio file."""
    temp_dir = os.path.join(ROOT, "data", "raw", "enrollments")
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, f"{speaker_name}.wav")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    from backend.services import audio_service
    success = audio_service.enroll_speaker(speaker_name, file_path)
    if success:
        return {"status": "success", "message": f"Enrolled speaker {speaker_name}"}
    else:
        return {"status": "error", "message": "Failed to extract speaker embedding"}


# ==============================================================================
# POST /audio/process — Process an audio segment: transcribe + verify + orchestrate
# ==============================================================================
@router.post("/audio/process")
async def process_audio(
    session_id: str = Form("default"),
    file: UploadFile = File(...),
    db: DBSession = Depends(get_db)
):
    """
    Process an uploaded audio segment:
    1. Resample and extract speaker verification embedding to match identity
    2. Transcribe speech to text using faster-whisper
    3. Feed resulting transcript into Orchestrator pipeline
    """
    auto_enroll_samples()

    temp_dir = os.path.join(ROOT, "data", "raw", "uploads")
    os.makedirs(temp_dir, exist_ok=True)
    
    import uuid
    filename = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        from backend.services import audio_service
        speaker = audio_service.identify_speaker(file_path)

        text, avg_prob = audio_service.transcribe_audio(file_path)
        if not text:
            return {"status": "warning", "message": "No transcription generated from audio."}

        orch = await get_orchestrator()
        
        db_session = db.query(models.Session).filter_by(session_id=session_id).first()
        if not db_session:
            db_session = models.Session(session_id=session_id)
            db.add(db_session)
            db.commit()

        transcript = models.Transcript(
            session_id=session_id,
            speaker=speaker,
            segment_text=text,
        )
        db.add(transcript)
        db.commit()
        db.refresh(transcript)

        segment = TranscriptSegment(
            session_id=session_id,
            speaker=speaker,
            segment_text=text,
        )
        output = await orch.process_segment(segment)

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

        queue = get_live_queue(session_id)
        await queue.put(output.model_dump_json())

        return {
            "status": "success",
            "speaker": speaker,
            "transcription": text,
            "pipeline_output": output.model_dump()
        }
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


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

    disconnect_event = asyncio.Event()

    async def read_from_client():
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect as e:
            log.info("WS /live: client disconnected via read in session %s, code=%s, reason=%s", session_id, e.code, getattr(e, "reason", None))
        except Exception as e:
            log.error("WS /live: client read error: %s", e)
        finally:
            disconnect_event.set()

    reader_task = asyncio.create_task(read_from_client())

    try:
        if redis_client:
            pubsub = redis_client.pubsub()
            pubsub.subscribe(channel)
            try:
                while not disconnect_event.is_set():
                    msg = pubsub.get_message(ignore_subscribe_messages=True)
                    if msg and msg["type"] == "message":
                        await websocket.send_text(msg["data"].decode("utf-8"))
                    await asyncio.sleep(0.1)
            finally:
                pubsub.unsubscribe(channel)
        else:
            queue = get_live_queue(session_id)
            while not disconnect_event.is_set():
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=1.0)
                    await websocket.send_text(payload)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        log.info("WS /live: client disconnected via write in session %s", session_id)
    except Exception as e:
        log.error("WS /live: error: %s", e)
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass
        log.info("WS /live: cleanup for session %s", session_id)
