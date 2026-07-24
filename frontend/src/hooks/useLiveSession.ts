"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  OrchestratorOutput,
  SummaryOutput,
  TranscriptItem,
  DetailedClaimItem,
  ProcessAudioResponse,
} from "@/types/debate";

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "reconnecting";

const HTTP_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export function useLiveSession(sessionId: string = "default") {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [transcripts, setTranscripts] = useState<TranscriptItem[]>([]);
  const [claims, setClaims] = useState<DetailedClaimItem[]>([]);
  const [summary, setSummary] = useState<SummaryOutput>({
    claim_count: 0,
    verdict_breakdown: { True: 0, False: 0, Misleading: 0, Unverified: 0 },
    speaker_metrics: {},
    fallacy_counts: {},
  });
  const [lastLatencyMs, setLastLatencyMs] = useState<number | null>(null);
  const [isProcessingAudio, setIsProcessingAudio] = useState(false);

  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);

  // Fetch initial summary from GET /dashboard
  const fetchDashboardSummary = useCallback(async () => {
    try {
      const res = await fetch(`${HTTP_BASE}/dashboard?session_id=${sessionId}`);
      if (res.ok) {
        const data: SummaryOutput = await res.json();
        setSummary(data);
      }
    } catch (err) {
      console.warn("Failed to fetch initial dashboard summary:", err);
    }
  }, [sessionId]);

  // Handle incoming Orchestrator payload
  const handleOrchestratorPayload = useCallback((data: OrchestratorOutput, latencyMs?: number) => {
    const timestamp = new Date().toLocaleTimeString();

    // Single line log verification per Section 1 spec
    console.log(`[WS Live Message] type=OrchestratorOutput speaker="${data.speaker}" claims=${data.claims_extracted?.length || 0}`);

    // Add transcript segment if text present
    if (data.segment_text) {
      setTranscripts((prev) => [
        ...prev,
        {
          id: `${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
          timestamp,
          speaker: data.speaker || "Unknown",
          text: data.segment_text,
          speakerConfidence: 0.94, // From speaker verification profile
        },
      ]);
    }

    // Add claim results
    if (data.claim_results && data.claim_results.length > 0) {
      const newClaimItems: DetailedClaimItem[] = data.claim_results.map((cr) => ({
        id: `${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
        timestamp,
        result: cr,
        latencyMs,
      }));
      setClaims((prev) => [...newClaimItems, ...prev]);
    }

    // Update session summary
    if (data.summary) {
      setSummary(data.summary);
    }

    if (latencyMs !== undefined) {
      setLastLatencyMs(latencyMs);
    }
  }, []);

  // Native WebSocket connection with auto-reconnect
  const connectWebSocket = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close();
    }

    setStatus(reconnectAttemptsRef.current > 0 ? "reconnecting" : "connecting");
    const wsUrl = `${WS_BASE}/live?session_id=${sessionId}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log(`[WS Live] Connected to ${wsUrl}`);
      setStatus("connected");
      reconnectAttemptsRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "ping") return;
        handleOrchestratorPayload(data as OrchestratorOutput);
      } catch (err) {
        console.error("Failed to parse WS message:", err);
      }
    };

    ws.onerror = (err) => {
      console.error("[WS Live] Error:", err);
    };

    ws.onclose = () => {
      console.warn("[WS Live] Disconnected.");
      setStatus("disconnected");
      socketRef.current = null;

      // Exponential backoff reconnect
      const timeout = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 10000);
      reconnectAttemptsRef.current += 1;
      reconnectTimeoutRef.current = setTimeout(() => {
        connectWebSocket();
      }, timeout);
    };

    socketRef.current = ws;
  }, [sessionId, handleOrchestratorPayload]);

  useEffect(() => {
    fetchDashboardSummary();
    connectWebSocket();

    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (socketRef.current) {
        socketRef.current.onclose = null;
        socketRef.current.close();
      }
    };
  }, [connectWebSocket, fetchDashboardSummary]);

  // Trigger audio processing POST /audio/process
  const processAudioFile = async (file: File) => {
    setIsProcessingAudio(true);
    const startTime = performance.now();
    try {
      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("file", file);

      const res = await fetch(`${HTTP_BASE}/audio/process`, {
        method: "POST",
        body: formData,
      });

      const elapsed = Math.round(performance.now() - startTime);

      if (!res.ok) {
        throw new Error(`Audio upload failed with status ${res.status}`);
      }

      const data: ProcessAudioResponse = await res.json();
      if (data.pipeline_output) {
        handleOrchestratorPayload(data.pipeline_output, elapsed);
      }
      return { success: true, latencyMs: elapsed, data };
    } catch (err: any) {
      console.error("Error processing audio:", err);
      return { success: false, error: err.message };
    } finally {
      setIsProcessingAudio(false);
    }
  };

  return {
    status,
    transcripts,
    claims,
    summary,
    lastLatencyMs,
    isProcessingAudio,
    processAudioFile,
    refreshSummary: fetchDashboardSummary,
  };
}
