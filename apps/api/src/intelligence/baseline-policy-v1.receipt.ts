import type { BaselinePolicyReceipt } from './baseline-policy-v1.types';

export const BASELINE_POLICY_V1: BaselinePolicyReceipt = Object.freeze({
  version: 'baseline-policy-v1',
  retentionWindowDays: 31,
  baselineWindowDays: 28,
  recentWindowDays: 3,
  learningMinDistinctDays: 2,
  establishedMinDistinctDays: 7,
  staleAfterDays: 7,
  minimumDirectionalSamples: 2,
  directionThreshold: 0.35,
  magnitudeThresholds: {
    moderate: 0.8,
    large: 1.35,
  },
  conflictRatio: 0.35,
  reliabilityWeights: {
    WEAK: 0.5,
    STANDARD: 1,
    STRONG: 1.5,
  },
  confidenceAgreementBonusPerSample: 1,
  confidenceThresholds: {
    mediumBaselineWeight: 5,
    mediumRecentEvidenceScore: 3,
    highBaselineWeight: 9,
    highRecentEvidenceScore: 6,
  },
});
