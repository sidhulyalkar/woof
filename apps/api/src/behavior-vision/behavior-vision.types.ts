export const BEHAVIOR_OBSERVATION_SCHEMA_VERSION = 'woof-behavior-observation-v1';
export const BEHAVIOR_PROFILE_SCHEMA_VERSION = 'woof-individual-behavior-profile-v1';

export const BEHAVIOR_DIMENSIONS = [
  'arousal',
  'body-tension',
  'social-orientation',
  'approach-tendency',
  'avoidance-tendency',
  'handler-engagement',
  'environment-engagement',
  'recovery',
] as const;

export type BehaviorDimension = (typeof BEHAVIOR_DIMENSIONS)[number];

export const BEHAVIOR_CONTEXTS = [
  'home',
  'street',
  'park',
  'dog-park',
  'trail',
  'daycare',
  'vet',
  'training-class',
  'vehicle',
  'other',
] as const;

export type BehaviorContext = (typeof BEHAVIOR_CONTEXTS)[number];

export const BEHAVIOR_PHASES = ['baseline', 'during-intervention', 'recovery'] as const;
export type BehaviorPhase = (typeof BEHAVIOR_PHASES)[number];

export const HANDLER_ACTIONS = [
  'none',
  'increase-distance',
  'decrease-distance',
  'loosen-leash',
  'tighten-leash',
  'single-cue',
  'repeated-cues',
  'find-it',
  'parallel-walk',
  'u-turn',
  'pause-and-observe',
  'allow-greeting',
  'end-interaction',
  'other',
] as const;

export type HandlerAction = (typeof HANDLER_ACTIONS)[number];

export type BehaviorEvidenceSource =
  'pose' | 'motion' | 'face' | 'audio' | 'interaction' | 'context' | 'owner';

export type BehaviorEvidence = {
  label: string;
  source: BehaviorEvidenceSource;
  confidence: number;
  startMs?: number;
  endMs?: number;
};

export type BehaviorDimensionEstimate = {
  dimension: BehaviorDimension;
  value: number;
  confidence: number;
  basis: string[];
};

export type BehaviorHypothesis = {
  id:
    | 'social-approach-with-arousal'
    | 'barrier-frustration-compatible-pattern'
    | 'avoidance-or-conflict-compatible-pattern'
    | 'play-compatible-pattern'
    | 'overarousal-compatible-pattern'
    | 'settled-observation'
    | 'insufficient-evidence';
  confidence: number;
  statement: string;
  supportingEvidence: string[];
  contradictoryEvidence: string[];
};

export type MediaQualityAssessment = {
  usable: boolean;
  confidence: number;
  issues: string[];
  recaptureInstructions: string[];
};

export type BehaviorVisionModelAnalysis = {
  schemaVersion: typeof BEHAVIOR_OBSERVATION_SCHEMA_VERSION;
  modelVersion: string;
  featureVersion: string;
  mediaQuality: MediaQualityAssessment;
  evidence: BehaviorEvidence[];
  dimensions: BehaviorDimensionEstimate[];
  hypotheses: BehaviorHypothesis[];
  observableSummary: string;
  uncertainty: string;
};

export type BehaviorObservationContext = {
  context: BehaviorContext;
  sessionKey?: string;
  phase: BehaviorPhase;
  handlerAction: HandlerAction;
  leashState: 'off-leash' | 'loose' | 'tight' | 'unknown';
  otherDogsPresent: boolean;
  otherDogDistanceMeters?: number;
  familiarDog?: boolean;
  audioAnalysisAllowed: boolean;
  ownerNote?: string;
};

export type StoredBehaviorObservation = {
  id: string;
  petId: string;
  createdAt: string;
  mediaType: 'image' | 'video';
  mediaSha256: string;
  context: BehaviorObservationContext;
  analysis: BehaviorVisionModelAnalysis;
  ownerFeedback?: {
    accurate: boolean;
    note?: string;
  };
};

export type PersonalizedDimensionBaseline = {
  dimension: BehaviorDimension;
  mean: number;
  confidence: number;
  sampleCount: number;
};

export type InterventionEffect = {
  action: HandlerAction;
  pairedSessions: number;
  arousalDelta: number | null;
  tensionDelta: number | null;
  engagementDelta: number | null;
  confidence: number;
};

export type IndividualBehaviorProfile = {
  schemaVersion: typeof BEHAVIOR_PROFILE_SCHEMA_VERSION;
  petId: string;
  sampleCount: number;
  contextsSeen: BehaviorContext[];
  baselines: PersonalizedDimensionBaseline[];
  interventionEffects: InterventionEffect[];
  personalizationConfidence: number;
  recommendation: {
    headline: string;
    explanation: string;
    nextSafeExperiment: string[];
    neverAutoRecommendGreeting: true;
  };
};
