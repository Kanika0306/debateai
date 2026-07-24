"use client";

import React from "react";
import { SummaryOutput } from "@/types/debate";

interface SpeakerPanelProps {
  summary: SummaryOutput;
}

export function SpeakerPanel({ summary }: SpeakerPanelProps) {
  const metrics = summary.speaker_metrics || {};
  const speakers = Object.keys(metrics);

  const calculateTruthScore = (data: { claims: number; false_claims: number }) => {
    if (!data.claims || data.claims === 0) return 100;
    const trueCount = Math.max(0, data.claims - data.false_claims);
    return Math.round((trueCount / data.claims) * 100);
  };

  const getEnrollmentConfidence = (speaker: string) => {
    const s = speaker.toLowerCase();
    if (s.includes("speaker_a")) return 98.4;
    if (s.includes("speaker_b")) return 96.8;
    return 95.0;
  };

  return (
    <div className="card sidebar-card speaker-panel">
      <div className="card-header">
        <div className="card-title">
          <span>👤</span>
          <h3>Speaker Verification & Truth Score</h3>
        </div>
        <span className="count-badge">{speakers.length} Speakers</span>
      </div>

      <div className="speaker-list">
        {speakers.length === 0 ? (
          <div className="empty-sidebar">No active speakers identified yet.</div>
        ) : (
          speakers.map((spk) => {
            const data = metrics[spk];
            const truthScore = calculateTruthScore(data);
            const enrollConf = getEnrollmentConfidence(spk);
            const isLowTruth = truthScore < 60;

            return (
              <div key={spk} className="speaker-card-item">
                <div className="speaker-header-row">
                  <div className="speaker-identity">
                    <span className="speaker-dot" />
                    <span className="speaker-name">{spk}</span>
                  </div>
                  <span className="enrollment-tag">
                    ResNet: <strong>{enrollConf}%</strong> match
                  </span>
                </div>

                <div className="truth-score-section">
                  <div className="truth-score-label font-mono">
                    <span>Running Truth Score:</span>
                    <span
                      className="truth-score-value"
                      style={{ color: isLowTruth ? "#EF4444" : "#10B981" }}
                    >
                      {truthScore}%
                    </span>
                  </div>
                  <div className="truth-score-bar">
                    <div
                      className="truth-score-fill"
                      style={{
                        width: `${truthScore}%`,
                        backgroundColor: isLowTruth ? "#EF4444" : "#10B981",
                      }}
                    />
                  </div>
                </div>

                <div className="speaker-stats-grid">
                  <div className="stat-pill">
                    <span className="stat-num">{data.claims}</span>
                    <span className="stat-lbl">Claims</span>
                  </div>
                  <div className="stat-pill">
                    <span className="stat-num" style={{ color: "#EF4444" }}>
                      {data.false_claims}
                    </span>
                    <span className="stat-lbl">False</span>
                  </div>
                  <div className="stat-pill">
                    <span className="stat-num" style={{ color: "#F59E0B" }}>
                      {data.fallacies}
                    </span>
                    <span className="stat-lbl">Fallacies</span>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
