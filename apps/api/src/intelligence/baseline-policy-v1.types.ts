export const SIGNAL_DIMENSIONS = [
  'APPETITE',
  'ENERGY',
  'BATHROOM_ROUTINE',
  'MOBILITY_COMFORT',
  'ENGAGEMENT_SOCIAL_COMFORT',
  'SLEEP_REST',
] as const;

export type SignalDimension = (typeof SIGNAL_DIMENSIONS)[number];

export const EVIDENCE_SOURCE_TYPES = [
  'OWNER_CHECKIN',
  'ACTIVITY',
  'COACHING',
  'HEALTH_LENS',
  'BEHAVIOR_VISION',
  'CONNECTOR',
] as const;

export type EvidenceSourceType = (typeof EVIDENCE_SOURCE_TYPES)[number];
export type EvidenceReliability = 'WEAK' | 'STANDARD' | 'STRONG';
export type DeltaBucket = -2 | -1 | 0 | 1 | 2;

export type DataState = 'INSUFFICIENT' | 'LEARNING' | 'ESTABLISHED' | 'STALE';
export type Direction = 'NEAR_BASELINE' | 'LOWER' | 'HIGHER' | 'MIXED' | 'UNAVAILABLE';
export type Magnitude = 'SMALL' | 'MODERATE' | 'LARGE' | 'UNAVAILABLE';
export type Confidence = 'LOW' | 'MEDIUM' | 'HIGH';

export type NormalizedObservation = {
  id: string;
  dedupeKey: string;
  dimension: SignalDimension;
  observedAt: string;
  localDate: string;
  deltaBucket: DeltaBucket;
  sourceType: EvidenceSourceType;
  reliability: EvidenceReliability;
  confidence: number;
  supersedesObservationId?: string;
};

export type EvidenceSourceSummary = {
  sourceType: EvidenceSourceType;
  samples: number;
  recentSamples: number;
};

export type BaselineWindow = {
  from: string;
  to: string;
};

export type BaselineDimension = {
  policyVersion: 'baseline-policy-v1';
  dimension: SignalDimension;
  dataState: DataState;
  direction: Direction;
  magnitude: Magnitude;
  confidence: Confidence;
  baselineSamples: number;
  recentSamples: number;
  baselineWindow: BaselineWindow | null;
  recentWindow: BaselineWindow | null;
  sources: EvidenceSourceSummary[];
  explanation: string;
};

export type BaselineSummary = {
  policyVersion: 'baseline-policy-v1';
  dimensions: Record<SignalDimension, BaselineDimension>;
};

export type BaselinePolicyReceipt = {
  version: 'baseline-policy-v1';
  retentionWindowDays: number;
  baselineWindowDays: number;
  recentWindowDays: number;
  learningMinDistinctDays: number;
  establishedMinDistinctDays: number;
  staleAfterDays: number;
  minimumDirectionalSamples: number;
  directionThreshold: number;
  magnitudeThresholds: {
    moderate: number;
    large: number;
  };
  conflictRatio: number;
  reliabilityWeights: Record<EvidenceReliability, number>;
  confidenceAgreementBonusPerSample: number;
  confidenceThresholds: {
    mediumBaselineWeight: number;
    mediumRecentEvidenceScore: number;
    highBaselineWeight: number;
    highRecentEvidenceScore: number;
  };
};
