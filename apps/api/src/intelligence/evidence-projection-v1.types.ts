import type {
  DeltaBucket,
  EvidenceReliability,
  NormalizedObservation,
  SignalDimension,
} from './baseline-policy-v1.types';

export const MEASURED_INTELLIGENCE_DIMENSIONS = [
  'ACTIVITY_LOAD',
  'RECOVERY_REST_PROXY',
  'TRAINING_COMFORT_SUCCESS',
] as const;

export type MeasuredIntelligenceDimension = (typeof MEASURED_INTELLIGENCE_DIMENSIONS)[number];
export type IntelligenceDimension = SignalDimension | MeasuredIntelligenceDimension;

export const QUALIFIED_PROJECTION_SOURCE_TYPES = ['OWNER_CHECKIN', 'ACTIVITY', 'COACHING'] as const;
export type QualifiedProjectionSourceType = (typeof QUALIFIED_PROJECTION_SOURCE_TYPES)[number];

export type ProjectionAuthority = 'BASELINE_ELIGIBLE' | 'CONTEXT_ONLY';

export type ProjectionObservationCandidate = {
  userId: string;
  petId: string;
  dimension: IntelligenceDimension;
  sourceType: QualifiedProjectionSourceType;
  sourceIdentity: string;
  sourceEventId?: string;
  sourceRecordId?: string;
  observedAt: string;
  localDate: string;
  deltaBucket?: DeltaBucket;
  numericValue?: number;
  unit?: string;
  confidence: number;
  reliability: EvidenceReliability;
  authority: ProjectionAuthority;
  normalizationVersion: 'evidence-normalization-v1';
  normalizationReason: string;
  context?: Record<string, unknown>;
  supersedesObservationId?: string;
};

export type PersistedProjectionObservation = ProjectionObservationCandidate & {
  id: string;
  ingestedAt: string;
  payloadHash: string;
  retractedAt: string | null;
  retractionReason: string | null;
};

export type ProjectionWriteReceipt = {
  observationId: string;
  payloadHash: string;
  duplicate: boolean;
};

export type ProjectionRetractionReceipt = {
  observationId: string;
  retracted: boolean;
  duplicate: boolean;
};

export type BaselineEvidenceRow = Pick<
  PersistedProjectionObservation,
  | 'id'
  | 'sourceIdentity'
  | 'dimension'
  | 'sourceType'
  | 'observedAt'
  | 'localDate'
  | 'deltaBucket'
  | 'confidence'
  | 'reliability'
>;

export function toBaselineObservation(row: BaselineEvidenceRow): NormalizedObservation {
  if (row.deltaBucket === undefined) {
    throw new Error('Baseline-eligible projection evidence requires a delta bucket');
  }

  return {
    id: row.id,
    dedupeKey: row.sourceIdentity,
    dimension: row.dimension as SignalDimension,
    observedAt: row.observedAt,
    localDate: row.localDate,
    deltaBucket: row.deltaBucket,
    sourceType: row.sourceType,
    reliability: row.reliability,
    confidence: row.confidence,
  };
}
