export type VerdictType = "True" | "False" | "Misleading" | "Unverified";

export interface ChunkMetadata {
  chunk_id: string;
  text: string;
  source_url: string;
  title: string;
  trust_tier: number;
  domain_topic: string;
  score: number;
}

export interface RetrievalOutput {
  claim: string;
  chunks: ChunkMetadata[];
  error?: string | null;
}

export interface FactVerificationOutput {
  claim: string;
  verdict: VerdictType | string;
  confidence: number;
  cited_chunks: string[];
  error?: string | null;
}

export interface FallacyOutput {
  text: string;
  fallacy_type: string;
  confidence: number;
  error?: string | null;
}

export interface JudgeOutput {
  claim: string;
  speaker: string;
  verdict: VerdictType;
  confidence: number;
  fallacy: string;
  cited_chunks: string[];
  action_required: boolean;
  error?: string | null;
}

export interface ClaimResult {
  judge_output: JudgeOutput;
  retrieval_output: RetrievalOutput;
  verification_output: FactVerificationOutput;
  fallacy_output: FallacyOutput;
}

export interface SummaryOutput {
  claim_count: number;
  verdict_breakdown: Record<VerdictType, number>;
  speaker_metrics: Record<string, { claims: number; fallacies: number; false_claims: number }>;
  fallacy_counts: Record<string, number>;
}

export interface OrchestratorOutput {
  session_id: string;
  speaker: string;
  segment_text: string;
  claims_extracted: string[];
  claim_results: ClaimResult[];
  summary: SummaryOutput;
  error?: string | null;
}

export interface ProcessAudioResponse {
  status: "success" | "warning" | "error";
  speaker: string;
  transcription: string;
  pipeline_output: OrchestratorOutput;
}

export interface TranscriptItem {
  id: string;
  timestamp: string;
  speaker: string;
  text: string;
  speakerConfidence?: number;
}

export interface DetailedClaimItem {
  id: string;
  timestamp: string;
  result: ClaimResult;
  latencyMs?: number;
}
