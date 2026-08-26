import {
  ADAPTIVE_PROFILE_SCHEMA_VERSION,
  normalizeQuestionAnswer,
  resolveCurrentProfileEvidence,
  type PersistedProfileEvidenceLike,
} from './adaptive-profile-v1';

function evidence(
  overrides: Partial<PersistedProfileEvidenceLike> & Pick<PersistedProfileEvidenceLike, 'id'>
): PersistedProfileEvidenceLike {
  return {
    dimension: 'DOG_SOCIAL_COMFORT',
    subject: 'DOG',
    state: 'KNOWN',
    value: ['SELECTIVELY_SOCIAL'],
    confidence: 1,
    provenance: 'OWNER_EXPLICIT',
    schemaVersion: ADAPTIVE_PROFILE_SCHEMA_VERSION,
    occurredAt: new Date('2026-08-25T18:00:00.000Z'),
    ...overrides,
  };
}

describe('adaptive-profile-v1', () => {
  it('keeps owner correction authoritative over newer inferred evidence', () => {
    const projected = resolveCurrentProfileEvidence([
      evidence({
        id: 'correction',
        provenance: 'OWNER_CORRECTION',
        value: ['PREFERS_SPACE'],
        occurredAt: new Date('2026-08-24T18:00:00.000Z'),
      }),
      evidence({
        id: 'inference',
        provenance: 'OUTCOME_INFERENCE',
        value: ['OFTEN_SOCIAL'],
        occurredAt: new Date('2026-08-25T18:00:00.000Z'),
      }),
    ]);

    expect(projected).toHaveLength(1);
    expect(projected[0]?.id).toBe('correction');
  });

  it('lets the newest correction explicitly clear an older known correction', () => {
    const projected = resolveCurrentProfileEvidence([
      evidence({
        id: 'older-known',
        provenance: 'OWNER_CORRECTION',
        state: 'KNOWN',
        confidence: 1,
        occurredAt: new Date('2026-08-24T18:00:00.000Z'),
      }),
      evidence({
        id: 'newer-unknown',
        provenance: 'OWNER_CORRECTION',
        state: 'UNKNOWN',
        value: null,
        confidence: 0,
        occurredAt: new Date('2026-08-25T18:00:00.000Z'),
      }),
    ]);

    expect(projected[0]?.id).toBe('newer-unknown');
    expect(projected[0]?.state).toBe('UNKNOWN');
  });

  it('ignores evidence from another schema version or mismatched subject', () => {
    const projected = resolveCurrentProfileEvidence([
      evidence({ id: 'future', schemaVersion: 'adaptive-profile-v2' }),
      evidence({ id: 'wrong-subject', subject: 'OWNER' }),
    ]);

    expect(projected).toEqual([]);
  });

  it('canonicalizes multi-select answers independently of click order', () => {
    expect(
      normalizeQuestionAnswer('profile-owner-goals-v1', 'ANSWERED', [
        'TRAINING',
        'MORE_ADVENTURES',
        'TRAINING',
      ])
    ).toEqual({
      dimension: 'OWNER_GOALS',
      values: ['MORE_ADVENTURES', 'TRAINING'],
    });
  });

  it('does not convert skip or not-sure into a preference value', () => {
    expect(normalizeQuestionAnswer('profile-dog-energy-v1', 'SKIPPED', undefined)).toEqual({
      dimension: 'DOG_ENERGY_PATTERN',
      values: null,
    });
    expect(normalizeQuestionAnswer('profile-dog-energy-v1', 'NOT_SURE', undefined)).toEqual({
      dimension: 'DOG_ENERGY_PATTERN',
      values: null,
    });
  });
});
