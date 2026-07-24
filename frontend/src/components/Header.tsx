"use client";

import React, { useRef } from "react";
import { ConnectionStatus } from "@/hooks/useLiveSession";

interface HeaderProps {
  status: ConnectionStatus;
  sessionId: string;
  onAudioUpload: (file: File) => void;
  isProcessingAudio: boolean;
  lastLatencyMs: number | null;
}

export function Header({
  status,
  sessionId,
  onAudioUpload,
  isProcessingAudio,
  lastLatencyMs,
}: HeaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const statusConfig = {
    connected: { label: "LIVE STREAM CONNECTED", color: "#10B981", pulse: true },
    connecting: { label: "CONNECTING...", color: "#F59E0B", pulse: true },
    reconnecting: { label: "RECONNECTING...", color: "#F59E0B", pulse: true },
    disconnected: { label: "OFFLINE / DISCONNECTED", color: "#EF4444", pulse: false },
  };

  const currentStatus = statusConfig[status] || statusConfig.disconnected;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onAudioUpload(e.target.files[0]);
    }
  };

  return (
    <header className="header-bar">
      <div className="header-brand">
        <div className="brand-badge">
          <span className="brand-icon">🎙️</span>
          <span className="brand-tag">PHASE 3 ACTIVE</span>
        </div>
        <h1 className="brand-title">DEBATE-AI</h1>
        <span className="brand-subtitle">Real-Time Fact Checker & Logical Fallacy Engine</span>
      </div>

      <div className="header-controls">
        <div className="session-tag">
          <span className="label">SESSION:</span>
          <span className="value">{sessionId}</span>
        </div>

        <div className="status-indicator">
          <span
            className={`status-dot ${currentStatus.pulse ? "pulse" : ""}`}
            style={{ backgroundColor: currentStatus.color }}
          />
          <span className="status-label" style={{ color: currentStatus.color }}>
            {currentStatus.label}
          </span>
        </div>

        {lastLatencyMs !== null && (
          <div className="latency-badge">
            <span className="label">LATENCY:</span>
            <span className="value">{lastLatencyMs} ms</span>
          </div>
        )}

        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="audio/*"
          style={{ display: "none" }}
        />

        <button
          className="btn-upload"
          onClick={() => fileInputRef.current?.click()}
          disabled={isProcessingAudio}
        >
          {isProcessingAudio ? (
            <>
              <span className="spinner" /> Processing Audio...
            </>
          ) : (
            <>
              <span>⚡</span> Upload WAV / Audio Segment
            </>
          )}
        </button>
      </div>
    </header>
  );
}
