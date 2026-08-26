import type { SignalDimension } from './baseline-policy-v1.types';
import { EVIDENCE_NORMALIZATION_V1 } from './evidence-normalization-v1.receipt';
import type {
  ProjectionObservationCandidate,
  QualifiedProjectionSourceType,
} from './evidence-projection-v1.types';

export type OwnerCheckinChoice = 'LESS' | 'USUAL' | 'MORE' | 'UNSURE';

const LOCAL_DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

function assertCommon(input: {
  userId: string;
  petId: string;
  observedAt: string;
  localDate: string;
  confidence: number;
}) {
  if (!input.userId || !input.petId)
    throw new Error('Evidence normalization requires user and pet IDs');
  if (!Number.isFinite(Date.parse(input.observedAt))) {
    throw new Error('Evidence normalization requires a valid observedAt timestamp');
  }
  if (!LOCAL_DATE_PATTERN.test(input.localDate)) {
    throw new Error('Evidence normalization requires an upstream-normalized localDate');
  }
  if (!Number.isFinite(input.confidence) || input.confidence <= 0 || input.confidence > 1) {
    throw new Error('Evidence normalization confidence must be in (0, 1]');
  }
}

function sourceIdentity(parts: readonly string[]) {
  const identity = parts.join(':');
  if (identity.length > EVIDENCE_NORMALIZATION_V1.maxSourceIdentityLength) {
    throw new Error('Evidence normalization source identity is too long');
  }
  return identity;
}

export function normalizeOwnerCheckinObservation(input: {
  userId: string;
  petId: string;
  careEventId: string;
  dimension: SignalDimension;
  choice: OwnerCheckinChoice;
  observedAt: string;
  localDate: string;
  confidence?: number;
  supersedesObservationId?: string;
}): ProjectionObservationCandidate | null {
  const confidence = input.confidence ?? 0.8;
  assertCommon({ ...input, confidence });

  if (input.choice === 'UNSURE') return null;

  return {
    userId: input.userId,
    petId: input.petId,
    dimension: input.dimension,
    sourceType: 'OWNER_CHECKIN',
    sourceIdentity: sourceIdentity(['care-event', input.careEventId, input.dimension]),
    sourceEventId: input.careEventId,
    observedAt: input.observedAt,
    localDate: input.localDate,
    deltaBucket: EVIDENCE_NORMALIZATION_V1.ownerChoiceBuckets[input.choice],
    confidence,
    reliability: 'STANDARD',
    authority: 'BASELINE_ELIGIBLE',
    normalizationVersion: EVIDENCE_NORMALIZATION_V1.version,
    normalizationReason:
      'Owner Daily Signals semantic choice normalized directly to a bounded relative bucket.',
    context: { choice: input.choice },
    ...(input.supersedesObservationId
      ? { supersedesObservationId: input.supersedesObservationId }
      : {}),
  };
}

export function normalizeActivityMeasurement(input: {
  userId: string;
  petId: string;
  activityId: string;
  dimension: 'ACTIVITY_LOAD' | 'RECOVERY_REST_PROXY';
  numericValue: number;
  unit: 'minutes' | 'load_units';
  observedAt: string;
  localDate: string;
  confidence?: number;
}): ProjectionObservationCandidate {
  const confidence = input.confidence ?? 0.78;
  assertCommon({ ...input, confidence });
  if (!Number.isFinite(input.numericValue) || input.numericValue < 0) {
    throw new Error('Activity measurement must be a finite non-negative value');
  }

  return {
    userId: input.userId,
    petId: input.petId,
    dimension: input.dimension,
    sourceType: 'ACTIVITY',
    sourceIdentity: sourceIdentity(['activity', input.activityId, input.dimension]),
    sourceRecordId: input.activityId,
    observedAt: input.observedAt,
    localDate: input.localDate,
    numericValue: input.numericValue,
    unit: input.unit,
    confidence,
    reliability: 'STANDARD',
    authority: 'CONTEXT_ONLY',
    normalizationVersion: EVIDENCE_NORMALIZATION_V1.version,
    normalizationReason:
      'Activity measurement preserved as context-only numeric evidence; no health-like relative bucket is inferred.',
    context: {},
  };
}

export function normalizeCoachingObservation(input: {
  userId: string;
  petId: string;
  coachingRecordId: string;
  comfortSuccess: number;
  observedAt: string;
  localDate: string;
  confidence?: number;
}): ProjectionObservationCandidate {
  const confidence = input.confidence ?? 0.72;
  assertCommon({ ...input, confidence });
  if (
    !Number.isFinite(input.comfortSuccess) ||
    input.comfortSuccess < 0 ||
    input.comfortSuccess > 1
  ) {
    throw new Error('Coaching comfort/success measurement must be in [0, 1]');
  }

  return {
    userId: input.userId,
    petId: input.petId,
    dimension: 'TRAINING_COMFORT_SUCCESS',
    sourceType: 'COACHING',
    sourceIdentity: sourceIdentity([
      'coaching',
      input.coachingRecordId,
      'TRAINING_COMFORT_SUCCESS',
    ]),
    sourceRecordId: input.coachingRecordId,
    observedAt: input.observedAt,
    localDate: input.localDate,
    numericValue: input.comfortSuccess,
    unit: 'ratio',
    confidence,
    reliability: 'STANDARD',
    authority: 'CONTEXT_ONLY',
    normalizationVersion: EVIDENCE_NORMALIZATION_V1.version,
    normalizationReason:
      'Coaching comfort/success is preserved as context-only evidence and cannot directly alter baseline health-like dimensions.',
    context: {},
  };
}

export function isQualifiedProjectionSourceType(
  value: string
): value is QualifiedProjectionSourceType {
  return (
    EVIDENCE_NORMALIZATION_V1.evidenceMappings.SELF_REPORT.sourceType === value ||
    EVIDENCE_NORMALIZATION_V1.evidenceMappings.ACTIVITY.sourceType === value ||
    EVIDENCE_NORMALIZATION_V1.evidenceMappings.COACH.sourceType === value
  );
}
