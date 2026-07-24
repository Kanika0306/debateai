"use client";

import React from "react";
import { DetailedClaimItem, VerdictType } from "@/types/debate";

interface TimelineProps {
  claims: DetailedClaimItem[];
}

export function Timeline({ claims }: TimelineProps) {
  const verdictColors: Record<VerdictType, string> = {
    True: "#10B981",
    False: "#EF4444",
    Misleading: "#F59E0B",
    Unverified: "#8B5CF6",
  };

  // Reverse chronological -> chronological for timeline display
  const timelineItems = [...claims].reverse();

  return (
    <div className="card sidebar-card timeline-card">
      <div className="card-header">
        <div className="card-title">
          <span>📈</span>
          <h3>Claims Timeline</h3>
        </div>
        <span className="count-badge">{claims.length} Events</span>
      </div>

      <div className="timeline-container">
        {timelineItems.length === 0 ? (
          <div className="empty-sidebar">No claims detected in timeline yet.</div>
        ) : (
          <div className="timeline-nodes-list">
            {timelineItems.map((item, index) => {
              const verdict = item.result.judge_output.verdict;
              const color = verdictColors[verdict] || verdictColors.Unverified;
              const claimSnippet =
                item.result.judge_output.claim.length > 50
                  ? `${item.result.judge_output.claim.substring(0, 50)}...`
                  : item.result.judge_output.claim;

              return (
                <div key={item.id || index} className="timeline-node-item">
                  <div className="node-marker-column">
                    <span className="node-dot" style={{ backgroundColor: color, boxShadow: `0 0 8px ${color}` }} />
                    {index < timelineItems.length - 1 && <span className="node-line" />}
                  </div>
                  <div className="node-content">
                    <div className="node-header">
                      <span className="node-speaker">{item.result.judge_output.speaker}</span>
                      <span className="node-time">{item.timestamp}</span>
                    </div>
                    <p className="node-claim-preview">"{claimSnippet}"</p>
                    <span className="node-verdict-tag" style={{ color, borderColor: color }}>
                      {verdict}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
