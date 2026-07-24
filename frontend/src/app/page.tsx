"use client";

import React from "react";
import { useLiveSession } from "@/hooks/useLiveSession";
import { Header } from "@/components/Header";
import { TranscriptFeed } from "@/components/TranscriptFeed";
import { ClaimCard } from "@/components/ClaimCard";
import { FallacyAlert } from "@/components/FallacyAlert";
import { SpeakerPanel } from "@/components/SpeakerPanel";
import { SessionStats } from "@/components/SessionStats";
import { Timeline } from "@/components/Timeline";

export default function DashboardPage() {
  const sessionId = "default";
  const {
    status,
    transcripts,
    claims,
    summary,
    lastLatencyMs,
    isProcessingAudio,
    processAudioFile,
  } = useLiveSession(sessionId);

  return (
    <div className="dashboard-root">
      <Header
        status={status}
        sessionId={sessionId}
        onAudioUpload={processAudioFile}
        isProcessingAudio={isProcessingAudio}
        lastLatencyMs={lastLatencyMs}
      />

      <main className="dashboard-grid">
        {/* Left/Main Column: Live Transcripts & Extracted Claims */}
        <div className="main-column">
          <TranscriptFeed transcripts={transcripts} />

          <section className="claims-section">
            <div className="claims-section-title">
              <h2>Verified Claims & Fallacies</h2>
              <span className="count-badge">{claims.length} Extracted Claims</span>
            </div>

            <div className="claims-list">
              {claims.length === 0 ? (
                <div className="card empty-state">
                  <span className="empty-icon">🛡️</span>
                  <p>No claims extracted yet. Upload an audio segment or speak to process live claims.</p>
                </div>
              ) : (
                claims.map((item) => (
                  <React.Fragment key={item.id}>
                    {/* Surfaced Fallacy Alert if flagged */}
                    <FallacyAlert item={item} />
                    {/* Surfaced Claim Card */}
                    <ClaimCard item={item} />
                  </React.Fragment>
                ))
              )}
            </div>
          </section>
        </div>

        {/* Right Sidebar: Speaker Panel, Session Stats & Claims Timeline */}
        <aside className="sidebar-column">
          <SessionStats summary={summary} lastLatencyMs={lastLatencyMs} />
          <SpeakerPanel summary={summary} />
          <Timeline claims={claims} />
        </aside>
      </main>
    </div>
  );
}
