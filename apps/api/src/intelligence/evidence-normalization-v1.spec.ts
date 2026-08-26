import { createHash } from 'node:crypto';
import type { EvidenceType } from '../care-events/care-event.types';
import {
  isQualifiedProjectionSourceType,
  normalizeActivityMeasurement,
  normalizeCoachingObservation,
  normalizeOwnerCheckinObservation,
} from './evidence-normalization-v1';
import {
  EVIDENCE_NORMALIZATION_V1,
  qualifiedEvidenceMapping,
} from './evidence-normalization-v1.receipt';

const common = {
  userId: 'user-1',
  petId: 'pet-1',
  observedAt: '2026-08-25T18:00:00.000Z',
  localDate: '2026-08-25',
};

describe('evidence-normalization-v1 receipt', () => {
  it('pins the complete normalization authority contract', () => {
    expect(EVIDENCE_NORMALIZATION_V1).toEqual({
      version: 'evidence-normalization-v1',
      ownerChoiceBuckets: { LESS: -1, USUAL: 0, MORE: 1 },
      evidenceMappings: {
        SELF_REPORT: { sourceType: 'OWNER_CHECKIN', authority: 'BASELINE_ELIGIBLE' },
        ACTIVITY: { sourceType: 'ACTIVITY', authority: 'CONTEXT_ONLY' },
        COACH: { sourceType: 'COACHING', authority: 'CONTEXT_ONLY' },
      },
      maxContextBytes: 4096,
      maxNormalizationReasonLength: 512,
      maxSourceIdentityLength: 256,
      baselineDimensions: [
        'APPETITE',
        'ENERGY',
        'BATHROOM_ROUTINE',
        'MOBILITY_COMFORT',
        'ENGAGEMENT_SOCIAL_COMFORT',
        'SLEEP_REST',
      ],
      contextOnlyDimensions: ['ACTIVITY_LOAD', 'RECOVERY_REST_PROXY', 'TRAINING_COMFORT_SUCCESS'],
    });

    const fingerprint = createHash('sha256')
      .update(JSON.stringify(EVIDENCE_NORMALIZATION_V1))
      .digest('hex');
    expect(fingerprint).toBe('5c4262303923f9499bd08e6f34d5b2a7890c1d1839aa1933da5a6e8f4a8825f0');
  });

  it('fails closed for evidence classes that are not qualified in Release 1', () => {
    const unqualified: EvidenceType[] = [
      'BEHAVIOR_VISION',
      'LOCATION',
      'MEDIA',
      'CLINIC',
      'WEARABLE',
    ];
    for (const evidenceType of unqualified) {
      expect(qualifiedEvidenceMapping(evidenceType)).toBeNull();
    }
  });
});

describe('evidence-normalization-v1 adapters', () => {
  it('maps owner semantic choices directly to bounded baseline-eligible buckets', () => {
    const less = normalizeOwnerCheckinObservation({
      ...common,
      careEventId: 'event-1',
      dimension: 'APPETITE',
      choice: 'LESS',
    });
    const usual = normalizeOwnerCheckinObservation({
      ...common,
      careEventId: 'event-2',
      dimension: 'ENERGY',
      choice: 'USUAL',
    });
    const more = normalizeOwnerCheckinObservation({
      ...common,
      careEventId: 'event-3',
      dimension: 'SLEEP_REST',
      choice: 'MORE',
    });

    expect(less).toMatchObject({
      sourceType: 'OWNER_CHECKIN',
      authority: 'BASELINE_ELIGIBLE',
      deltaBucket: -1,
      sourceEventId: 'event-1',
    });
    expect(usual?.deltaBucket).toBe(0);
    expect(more?.deltaBucket).toBe(1);
  });

  it('treats unsure as missing evidence rather than normal evidence', () => {
    expect(
      normalizeOwnerCheckinObservation({
        ...common,
        careEventId: 'event-unsure',
        dimension: 'APPETITE',
        choice: 'UNSURE',
      })
    ).toBeNull();
  });

  it('keeps activity measurements context-only without inventing a relative bucket', () => {
    const result = normalizeActivityMeasurement({
      ...common,
      activityId: 'activity-1',
      dimension: 'ACTIVITY_LOAD',
      numericValue: 42,
      unit: 'minutes',
    });

    expect(result).toMatchObject({
      sourceType: 'ACTIVITY',
      authority: 'CONTEXT_ONLY',
      numericValue: 42,
      unit: 'minutes',
    });
    expect(result.deltaBucket).toBeUndefined();
  });

  it('keeps coaching comfort/success context-only', () => {
    const result = normalizeCoachingObservation({
      ...common,
      coachingRecordId: 'coach-1',
      comfortSuccess: 0.75,
    });

    expect(result).toMatchObject({
      dimension: 'TRAINING_COMFORT_SUCCESS',
      sourceType: 'COACHING',
      authority: 'CONTEXT_ONLY',
      numericValue: 0.75,
      unit: 'ratio',
    });
    expect(result.deltaBucket).toBeUndefined();
  });

  it('rejects malformed time, local-day and confidence inputs', () => {
    expect(() =>
      normalizeOwnerCheckinObservation({
        ...common,
        observedAt: 'nope',
        careEventId: 'event-bad-time',
        dimension: 'APPETITE',
        choice: 'USUAL',
      })
    ).toThrow('valid observedAt');

    expect(() =>
      normalizeActivityMeasurement({
        ...common,
        localDate: '08/25/2026',
        activityId: 'activity-bad-date',
        dimension: 'ACTIVITY_LOAD',
        numericValue: 10,
        unit: 'minutes',
      })
    ).toThrow('upstream-normalized localDate');

    expect(() =>
      normalizeCoachingObservation({
        ...common,
        coachingRecordId: 'coach-bad-confidence',
        comfortSuccess: 0.5,
        confidence: 2,
      })
    ).toThrow('confidence must be in (0, 1]');
  });

  it('recognizes only the three qualified projection source classes', () => {
    expect(isQualifiedProjectionSourceType('OWNER_CHECKIN')).toBe(true);
    expect(isQualifiedProjectionSourceType('ACTIVITY')).toBe(true);
    expect(isQualifiedProjectionSourceType('COACHING')).toBe(true);
    expect(isQualifiedProjectionSourceType('HEALTH_LENS')).toBe(false);
    expect(isQualifiedProjectionSourceType('BEHAVIOR_VISION')).toBe(false);
  });
});
