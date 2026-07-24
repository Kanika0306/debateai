"""
agents/orchestrator.py — Wires all agents into the live debate pipeline.

Flow per segment:
  transcript_segment
    -> ClaimExtractionAgent (extract claims)
    -> [parallel per claim: RetrievalAgent + FactVerificationAgent + FallacyAgent]
    -> JudgeAgent (resolve per claim)
    -> SummaryAgent (update session totals)
    -> publish to Redis pub/sub channel  debate:{session_id}:live
"""
import asyncio
import json
import logging
import os
from typing import Optional

from agents.schemas import (
    TranscriptSegment,
    ClaimExtractionInput,
    RetrievalInput,
    FactVerificationInput,
    FallacyInput,
    JudgeInput,
    SummaryInput,
    OrchestratorOutput,
    ClaimResult,
    SummaryOutput,
)
from agents.claim_extraction_agent import ClaimExtractionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.fact_verification_agent import FactVerificationAgent
from agents.fallacy_agent import FallacyAgent
from agents.judge_agent import JudgeAgent
from agents.summary_agent import SummaryAgent

log = logging.getLogger(__name__)


class Orchestrator:
    """
    Top-level pipeline controller. Holds agent instances and a session-scoped
    SummaryAgent. Processes one TranscriptSegment at a time.
    """

    def __init__(self, session_id: str = "default", use_redis: bool = True):
        self.session_id = session_id

        # Initialize agents (retrieval is heavy — loaded once)
        log.info("Orchestrator: initializing agents for session %s ...", session_id)
        self.claim_agent = ClaimExtractionAgent()
        self.fallacy_agent = FallacyAgent()
        self.judge_agent = JudgeAgent()
        self.summary_agent = SummaryAgent()

        # Retrieval agent is expensive (loads models) — lazy init
        self._retrieval_agent: Optional[RetrievalAgent] = None
        self._fact_agent = FactVerificationAgent()

        # Redis pub/sub (optional — falls back to in-memory queue)
        self._redis = None
        self._mem_queue: asyncio.Queue = asyncio.Queue()
        if use_redis:
            self._init_redis()

    def _init_redis(self):
        """Best-effort Redis connection. Falls back silently on failure."""
        try:
            import redis as redis_lib
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            self._redis = redis_lib.Redis.from_url(redis_url)
            self._redis.ping()
            log.info("Orchestrator: connected to Redis at %s", redis_url)
        except Exception as e:
            log.warning(
                "Orchestrator: Redis unavailable (%s). Using in-memory queue.", e
            )
            self._redis = None

    @property
    def retrieval_agent(self) -> RetrievalAgent:
        if self._retrieval_agent is None:
            self._retrieval_agent = RetrievalAgent()
        return self._retrieval_agent

    async def process_segment(self, segment: TranscriptSegment) -> OrchestratorOutput:
        """
        Main entry point: process one transcript segment end-to-end.
        """
        log.info(
            "Orchestrator: processing segment from speaker=%s, len=%d chars",
            segment.speaker, len(segment.segment_text),
        )

        # ── Step 1: Extract claims ──
        extraction_input = ClaimExtractionInput(
            segment_text=segment.segment_text,
            speaker=segment.speaker,
        )
        extraction_output = await self.claim_agent.run_with_timeout(
            extraction_input, timeout=10.0
        )

        claims = extraction_output.claims
        if not claims:
            log.info("Orchestrator: no claims extracted from segment.")
            return OrchestratorOutput(
                session_id=segment.session_id,
                speaker=segment.speaker,
                segment_text=segment.segment_text,
                claims_extracted=[],
                claim_results=[],
                summary=await self._get_summary([]),
                error=extraction_output.error,
            )

        log.info("Orchestrator: extracted %d claims: %s", len(claims), claims)

        # ── Step 2: Process each claim in parallel ──
        tasks = [
            self._process_single_claim(claim, segment.speaker)
            for claim in claims
        ]
        claim_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions from gather
        final_results = []
        for i, result in enumerate(claim_results):
            if isinstance(result, Exception):
                log.error("Orchestrator: claim %d failed: %s", i, result)
                # Create a fallback result
                from agents.schemas import (
                    JudgeOutput, RetrievalOutput,
                    FactVerificationOutput, FallacyOutput,
                )
                fallback = ClaimResult(
                    judge_output=JudgeOutput(
                        claim=claims[i], speaker=segment.speaker,
                        verdict="Unverified", confidence=0.0,
                        fallacy="no fallacy", error=str(result),
                    ),
                    retrieval_output=RetrievalOutput(claim=claims[i], chunks=[]),
                    verification_output=FactVerificationOutput(
                        claim=claims[i], verdict="Unverified",
                        confidence=0.0, cited_chunks=[],
                    ),
                    fallacy_output=FallacyOutput(
                        text=claims[i], fallacy_type="no fallacy", confidence=0.0,
                    ),
                )
                final_results.append(fallback)
            else:
                final_results.append(result)

        # ── Step 3: Update summary ──
        judge_outputs = [r.judge_output for r in final_results]
        summary = await self._get_summary(judge_outputs)

        # ── Step 4: Build output and publish ──
        output = OrchestratorOutput(
            session_id=segment.session_id,
            speaker=segment.speaker,
            segment_text=segment.segment_text,
            claims_extracted=claims,
            claim_results=final_results,
            summary=summary,
        )

        await self._publish(output)
        return output

    async def _process_single_claim(
        self, claim: str, speaker: str
    ) -> ClaimResult:
        """
        Per-claim pipeline: retrieve + verify + fallacy in parallel, then judge.
        """
        # Run retrieval, fallacy in parallel
        retrieval_task = self.retrieval_agent.run_with_timeout(
            RetrievalInput(claim=claim), timeout=15.0
        )
        fallacy_task = self.fallacy_agent.run_with_timeout(
            FallacyInput(text=claim), timeout=10.0
        )

        retrieval_output, fallacy_output = await asyncio.gather(
            retrieval_task, fallacy_task
        )

        # Verification depends on retrieval results
        verification_output = await self._fact_agent.run_with_timeout(
            FactVerificationInput(
                claim=claim,
                evidence=retrieval_output.chunks,
            ),
            timeout=10.0,
        )

        # Judge combines verification + fallacy
        judge_output = await self.judge_agent.run_with_timeout(
            JudgeInput(
                claim=claim,
                speaker=speaker,
                verification=verification_output,
                fallacy=fallacy_output,
            ),
            timeout=5.0,
        )

        return ClaimResult(
            judge_output=judge_output,
            retrieval_output=retrieval_output,
            verification_output=verification_output,
            fallacy_output=fallacy_output,
        )

    async def _get_summary(self, new_verdicts):
        """Update the session summary with new judge outputs."""
        if not new_verdicts:
            return SummaryOutput(
                claim_count=self.summary_agent._claim_count,
                verdict_breakdown=dict(self.summary_agent._verdict_breakdown),
                speaker_metrics={
                    k: dict(v) for k, v in self.summary_agent._speaker_metrics.items()
                },
                fallacy_counts=dict(self.summary_agent._fallacy_counts),
            )
        return await self.summary_agent.run(SummaryInput(new_verdicts=new_verdicts))

    async def _publish(self, output: OrchestratorOutput):
        """Publish result to Redis pub/sub or in-memory queue."""
        channel = f"debate:{self.session_id}:live"
        payload = output.model_dump_json()

        if self._redis:
            try:
                self._redis.publish(channel, payload)
                log.info("Orchestrator: published to Redis channel %s", channel)
            except Exception as e:
                log.warning("Orchestrator: Redis publish failed: %s", e)
                await self._mem_queue.put(payload)
        else:
            await self._mem_queue.put(payload)
            log.info("Orchestrator: queued result in memory (Redis unavailable).")


# Standalone verification runner
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def test():
        orch = Orchestrator(session_id="test_session", use_redis=False)

        segment = TranscriptSegment(
            segment_text=(
                "The inflation rate in Chicago increased by 10 percent in 2019. "
                "Also, WHO says shingles affects 1.2 to 3.4 per 1000 people per year. "
                "Everyone knows nuclear power has caused millions of deaths, "
                "so we must switch to solar immediately."
            ),
            speaker="Alice",
            session_id="test_session",
        )

        print("Running standalone Orchestrator test...")
        output = await orch.process_segment(segment)
        print("\n=== Orchestrator Output ===")
        print(output.model_dump_json(indent=2))

    asyncio.run(test())
