"use client";

import React, { useState } from "react";
import { DetailedClaimItem, VerdictType } from "@/types/debate";

interface ClaimCardProps {
  item: DetailedClaimItem;
}

export function ClaimCard({ item }: ClaimCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const judge = item.result.judge_output;
  const retrieval = item.result.retrieval_output;
  const verification = item.result.verification_output;

  const verdictStyles: Record<VerdictType, { bg: string; border: string; text: string; label: string }> = {
    True: { bg: "rgba(16, 185, 129, 0.15)", border: "#10B981", text: "#10B981", label: "TRUE FACT" },
    False: { bg: "rgba(239, 68, 68, 0.15)", border: "#EF4444", text: "#EF4444", label: "FALSE assertion" },
    Misleading: { bg: "rgba(245, 158, 11, 0.15)", border: "#F59E0B", text: "#F59E0B", label: "MISLEADING" },
    Unverified: { bg: "rgba(139, 92, 246, 0.15)", border: "#8B5CF6", text: "#8B5CF6", label: "UNVERIFIED" },
  };

  const style = verdictStyles[judge.verdict] || verdictStyles.Unverified;
  const confPct = Math.round(judge.confidence * 100);

  return (
    <div className="claim-card" style={{ borderLeftColor: style.border }}>
      <div className="claim-card-header">
        <div className="claim-speaker-meta">
          <span className="speaker-tag">{judge.speaker}</span>
          <span className="claim-timestamp">{item.timestamp}</span>
          {item.latencyMs && <span className="latency-tag">{item.latencyMs}ms pipeline</span>}
        </div>
        <div className="verdict-badge" style={{ backgroundColor: style.bg, borderColor: style.border, color: style.text }}>
          <span className="verdict-dot" style={{ backgroundColor: style.border }} />
          <span>{style.label}</span>
        </div>
      </div>

      <div className="claim-body">
        <p className="claim-text">"{judge.claim}"</p>
      </div>

      {/* Confidence Bar */}
      <div className="confidence-section">
        <div className="confidence-label-row">
          <span>Verdict Confidence</span>
          <span className="confidence-val" style={{ color: style.text }}>{confPct}%</span>
        </div>
        <div className="confidence-track">
          <div
            className="confidence-fill"
            style={{ width: `${confPct}%`, backgroundColor: style.border }}
          />
        </div>
      </div>

      {/* Evidence Summary Line */}
      {retrieval.chunks && retrieval.chunks.length > 0 && (
        <div className="evidence-summary-line">
          <span className="evidence-icon">📚</span>
          <span className="evidence-source-title">
            Source: <strong>{retrieval.chunks[0].title || retrieval.chunks[0].source_url}</strong>
          </span>
          <span className="evidence-score-badge">
            Score: {(retrieval.chunks[0].score || 0).toFixed(2)}
          </span>
          <button
            className="btn-toggle-evidence"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? "Hide Source Text ▲" : `Expand ${retrieval.chunks.length} Evidence Chunks ▼`}
          </button>
        </div>
      )}

      {/* Expandable Source Text Panel */}
      {isExpanded && retrieval.chunks && (
        <div className="evidence-drawer">
          <h4 className="drawer-title">Cited RAG Chunks & Verification Context</h4>
          {retrieval.chunks.map((chunk, idx) => (
            <div key={chunk.chunk_id || idx} className="chunk-box">
              <div className="chunk-header">
                <span className="chunk-id">#{chunk.chunk_id}</span>
                <a href={chunk.source_url} target="_blank" rel="noreferrer" className="chunk-url">
                  {chunk.source_url || chunk.title}
                </a>
                <span className="chunk-tier">Trust Tier {chunk.trust_tier}</span>
              </div>
              <p className="chunk-text">{chunk.text}</p>
            </div>
          ))}
          {verification.error && (
            <div className="verification-error-box">
              <span>Verification Note: {verification.error}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
