"use client";

import React from "react";
import { DetailedClaimItem } from "@/types/debate";

interface FallacyAlertProps {
  item: DetailedClaimItem;
}

export function FallacyAlert({ item }: FallacyAlertProps) {
  const fallacy = item.result.fallacy_output;
  const judge = item.result.judge_output;

  // Filter out 'no fallacy'
  if (!fallacy || fallacy.fallacy_type === "no fallacy" || judge.fallacy === "no fallacy") {
    return null;
  }

  const fallacyType = fallacy.fallacy_type || judge.fallacy;
  const confidence = Math.round((fallacy.confidence || 0.85) * 100);

  return (
    <div className="fallacy-alert-banner">
      <div className="fallacy-alert-header">
        <div className="fallacy-title-group">
          <span className="fallacy-warning-icon">⚠️</span>
          <span className="fallacy-badge">LOGICAL FALLACY DETECTED</span>
          <span className="fallacy-type-tag">{fallacyType.toUpperCase()}</span>
        </div>
        <span className="fallacy-confidence">{confidence}% Confidence</span>
      </div>

      <div className="fallacy-body">
        <div className="flagged-span-box">
          <span className="span-label">FLAGGED SPAN:</span>
          <span className="span-quote">"{fallacy.text || judge.claim}"</span>
        </div>
        <div className="speaker-attribution">
          Attributed to <strong>{judge.speaker}</strong> at {item.timestamp}
        </div>
      </div>
    </div>
  );
}
