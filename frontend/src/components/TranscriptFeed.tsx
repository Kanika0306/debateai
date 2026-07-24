"use client";

import React, { useEffect, useRef } from "react";
import { TranscriptItem } from "@/types/debate";

interface TranscriptFeedProps {
  transcripts: TranscriptItem[];
}

export function TranscriptFeed({ transcripts }: TranscriptFeedProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcripts]);

  const getSpeakerColor = (speaker: string) => {
    const s = speaker.toLowerCase();
    if (s.includes("speaker_a") || s.includes("a")) return "#3B82F6"; // Blue
    if (s.includes("speaker_b") || s.includes("b")) return "#8B5CF6"; // Purple
    if (s.includes("speaker_c") || s.includes("c")) return "#EC4899"; // Pink
    return "#10B981"; // Emerald
  };

  return (
    <div className="card transcript-feed">
      <div className="card-header">
        <div className="card-title">
          <span>💬</span>
          <h2>Live Transcript Feed</h2>
        </div>
        <span className="count-badge">{transcripts.length} Segments</span>
      </div>

      <div className="transcript-scroll" ref={scrollRef}>
        {transcripts.length === 0 ? (
          <div className="empty-state">
            <span className="empty-icon">📡</span>
            <p>Waiting for live audio or transcription stream...</p>
          </div>
        ) : (
          transcripts.map((item) => {
            const color = getSpeakerColor(item.speaker);
            return (
              <div key={item.id} className="transcript-row">
                <div className="speaker-avatar" style={{ backgroundColor: `${color}20`, borderColor: color, color }}>
                  {item.speaker.substring(0, 2).toUpperCase()}
                </div>
                <div className="transcript-content">
                  <div className="transcript-meta">
                    <span className="speaker-name" style={{ color }}>{item.speaker}</span>
                    {item.speakerConfidence && (
                      <span className="confidence-chip">{(item.speakerConfidence * 100).toFixed(0)}% match</span>
                    )}
                    <span className="timestamp">{item.timestamp}</span>
                  </div>
                  <p className="transcript-text">{item.text}</p>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
