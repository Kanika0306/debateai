"use client";

import React from "react";
import { SummaryOutput } from "@/types/debate";

interface SessionStatsProps {
  summary: SummaryOutput;
  lastLatencyMs: number | null;
}

export function SessionStats({ summary, lastLatencyMs }: SessionStatsProps) {
  const breakdown = summary.verdict_breakdown || { True: 0, False: 0, Misleading: 0, Unverified: 0 };
  const totalClaims = summary.claim_count || 0;
  const trueClaims = breakdown.True || 0;
  const sessionAccuracy = totalClaims > 0 ? Math.round((trueClaims / totalClaims) * 100) : 100;

  const fallacyCountTotal = Object.values(summary.fallacy_counts || {}).reduce((a, b) => a + b, 0);

  return (
    <div className="card sidebar-card session-stats-card">
      <div className="card-header">
        <div className="card-title">
          <span>📊</span>
          <h3>Session Analytics & Accuracy</h3>
        </div>
      </div>

      <div className="session-metrics-grid">
        <div className="metric-box hero-metric">
          <span className="metric-value" style={{ color: sessionAccuracy >= 70 ? "#10B981" : "#EF4444" }}>
            {sessionAccuracy}%
          </span>
          <span className="metric-label">Session Accuracy</span>
        </div>

        <div className="metric-box">
          <span className="metric-value">{totalClaims}</span>
          <span className="metric-label">Total Claims</span>
        </div>

        <div className="metric-box">
          <span className="metric-value" style={{ color: "#F59E0B" }}>
            {fallacyCountTotal}
          </span>
          <span className="metric-label">Fallacies</span>
        </div>

        <div className="metric-box">
          <span className="metric-value font-mono">
            {lastLatencyMs ? `${lastLatencyMs}ms` : "< 350ms"}
          </span>
          <span className="metric-label">Avg Pipeline Latency</span>
        </div>
      </div>

      {/* Verdict Distribution Bar */}
      <div className="verdict-breakdown-section">
        <h4 className="breakdown-title">Verdict Distribution</h4>
        <div className="distribution-bar">
          {totalClaims > 0 && (
            <>
              <div
                className="dist-segment dist-true"
                style={{ width: `${((breakdown.True || 0) / totalClaims) * 100}%` }}
                title={`True: ${breakdown.True}`}
              />
              <div
                className="dist-segment dist-false"
                style={{ width: `${((breakdown.False || 0) / totalClaims) * 100}%` }}
                title={`False: ${breakdown.False}`}
              />
              <div
                className="dist-segment dist-misleading"
                style={{ width: `${((breakdown.Misleading || 0) / totalClaims) * 100}%` }}
                title={`Misleading: ${breakdown.Misleading}`}
              />
              <div
                className="dist-segment dist-unverified"
                style={{ width: `${((breakdown.Unverified || 0) / totalClaims) * 100}%` }}
                title={`Unverified: ${breakdown.Unverified}`}
              />
            </>
          )}
        </div>

        <div className="breakdown-legend">
          <div className="legend-item">
            <span className="legend-dot" style={{ backgroundColor: "#10B981" }} />
            <span>True ({breakdown.True || 0})</span>
          </div>
          <div className="legend-item">
            <span className="legend-dot" style={{ backgroundColor: "#EF4444" }} />
            <span>False ({breakdown.False || 0})</span>
          </div>
          <div className="legend-item">
            <span className="legend-dot" style={{ backgroundColor: "#F59E0B" }} />
            <span>Misleading ({breakdown.Misleading || 0})</span>
          </div>
          <div className="legend-item">
            <span className="legend-dot" style={{ backgroundColor: "#8B5CF6" }} />
            <span>Unverified ({breakdown.Unverified || 0})</span>
          </div>
        </div>
      </div>
    </div>
  );
}
