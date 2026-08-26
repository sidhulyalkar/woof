import { createHash } from 'node:crypto';
import {
  PROFILE_QUESTION_POLICY_V1,
  PROFILE_QUESTION_POLICY_V1_CANONICAL_RECEIPT,
  PROFILE_QUESTION_POLICY_V1_SHA256,
} from './profile-question-policy-v1.receipt';
import { selectProfileQuestion } from './profile-question-policy-v1';
import {
  PROFILE_DIMENSIONS,
  type ProfileDimension,
  type ProfileEvidence,
  type ProfileQuestionPolicyInput,
} from './profile-question-policy-v1.types';

const NOW = '2026-08-25T20:00:00.000Z';

function evidence(
  dimension: ProfileDimension,
  overrides: Partial<ProfileEvidence> = {}
): ProfileEvidence {
  return {
    dimension,
    state: 'KNOWN',
    confidence: 0.95,
    provenance: 'OWNER_EXPLICIT',
    updatedAt: '2026-08-25T18:00:00.000Z',
    ...overrides,
  };
}

function allKnown(except: readonly ProfileDimension[] = []): ProfileEvidence[] {
  return PROFILE_DIMENSIONS.filter((dimension) => !except.includes(dimension)).map((dimension) =>
    evidence(dimension)
  );
}

function input(overrides: Partial<ProfileQuestionPolicyInput> = {}): ProfileQuestionPolicyInput {
  return {
    profileEvidence: [],
    questionHistory: [],
    interactions: [],
    now: NOW,
    ...overrides,
  };
}

describe('profile-question-policy-v1', () => {
  it('pins the complete policy receipt to its SHA-256', () => {
    expect(JSON.parse(PROFILE_QUESTION_POLICY_V1_CANONICAL_RECEIPT)).toEqual(
      PROFILE_QUESTION_POLICY_V1
    );
    expect(
      createHash('sha256').update(PROFILE_QUESTION_POLICY_V1_CANONICAL_RECEIPT).digest('hex')
    ).toBe(PROFILE_QUESTION_POLICY_V1_SHA256);
  });

  it('starts sparse profiles with one high-value foundational question', () => {
    const question = selectProfileQuestion(input());

    expect(question?.id).toBe('profile-owner-goals-v1');
    expect(question?.reasonCodes).toContain('FOUNDATIONAL_COLD_START');
  });

  it('returns no question when every durable dimension is already known', () => {
    expect(selectProfileQuestion(input({ profileEvidence: allKnown() }))).toBeNull();
  });

  it('respects a recent skip without blocking the rest of the experience', () => {
    const question = selectProfileQuestion(
      input({
        questionHistory: [
          {
            questionId: 'profile-owner-goals-v1',
            askedAt: '2026-08-20T20:00:00.000Z',
            outcome: 'SKIPPED',
          },
        ],
      })
    );

    expect(question).not.toBeNull();
    expect(question?.id).not.toBe('profile-owner-goals-v1');
  });

  it('allows a skipped question to become eligible again after its cooldown', () => {
    const question = selectProfileQuestion(
      input({
        profileEvidence: allKnown(['OWNER_GOALS']),
        questionHistory: [
          {
            questionId: 'profile-owner-goals-v1',
            askedAt: '2026-08-10T20:00:00.000Z',
            outcome: 'SKIPPED',
          },
        ],
      })
    );

    expect(question?.id).toBe('profile-owner-goals-v1');
  });

  it('uses the most conservative cooldown outcome when duplicate history timestamps conflict', () => {
    const question = selectProfileQuestion(
      input({
        profileEvidence: allKnown(['OWNER_GOALS']),
        questionHistory: [
          {
            questionId: 'profile-owner-goals-v1',
            askedAt: '2026-08-16T20:00:00.000Z',
            outcome: 'ANSWERED',
          },
          {
            questionId: 'profile-owner-goals-v1',
            askedAt: '2026-08-16T20:00:00.000Z',
            outcome: 'SKIPPED',
          },
        ],
      })
    );

    expect(question).toBeNull();
  });

  it('does not turn one declined scent quest into a durable preference question', () => {
    const question = selectProfileQuestion(
      input({
        interactions: [
          {
            id: 'dismiss-1',
            occurredAt: '2026-08-25T18:00:00.000Z',
            kind: 'DISMISSED',
            questFamily: 'SCENT',
          },
        ],
      })
    );

    expect(question?.target.kind).toBe('PROFILE');
  });

  it('does not count a replayed interaction id as two independent dismissals', () => {
    const replayedDismissal = {
      id: 'dismiss-1',
      occurredAt: '2026-08-25T18:00:00.000Z',
      kind: 'DISMISSED' as const,
      questFamily: 'SCENT' as const,
    };
    const question = selectProfileQuestion(
      input({ interactions: [replayedDismissal, { ...replayedDismissal }] })
    );

    expect(question?.target.kind).toBe('PROFILE');
  });

  it('fails closed on a divergent replay sharing one interaction id', () => {
    const question = selectProfileQuestion(
      input({
        interactions: [
          {
            id: 'interaction-1',
            occurredAt: '2026-08-25T18:00:00.000Z',
            kind: 'DISMISSED',
            questFamily: 'SOCIAL',
          },
          {
            id: 'interaction-1',
            occurredAt: '2026-08-25T18:00:00.000Z',
            kind: 'COMPLETED',
            questFamily: 'SOCIAL',
            dogExperience: 'comfortable',
          },
        ],
      })
    );

    expect(question?.target.kind).toBe('PROFILE');
  });

  it('asks preference-versus-timing only after repeated related dismissals', () => {
    const question = selectProfileQuestion(
      input({
        interactions: [
          {
            id: 'dismiss-1',
            occurredAt: '2026-08-24T18:00:00.000Z',
            kind: 'DISMISSED',
            questFamily: 'SCENT',
          },
          {
            id: 'dismiss-2',
            occurredAt: '2026-08-25T18:00:00.000Z',
            kind: 'DISMISSED',
            questFamily: 'SCENT',
          },
        ],
      })
    );

    expect(question?.target).toEqual({ kind: 'QUEST_FAMILY_PREFERENCE', questFamily: 'SCENT' });
    expect(question?.reasonCodes).toContain('REPEATED_RELATED_DISMISSALS');
  });

  it('never treats safe opt-outs as evidence of a durable dislike', () => {
    const question = selectProfileQuestion(
      input({
        interactions: [
          {
            id: 'stop-1',
            occurredAt: '2026-08-24T18:00:00.000Z',
            kind: 'SAFE_OPT_OUT',
            questFamily: 'SOCIAL',
          },
          {
            id: 'stop-2',
            occurredAt: '2026-08-25T18:00:00.000Z',
            kind: 'SAFE_OPT_OUT',
            questFamily: 'SOCIAL',
          },
        ],
      })
    );

    expect(question?.target.kind).toBe('PROFILE');
  });

  it('can prefer one timely training-difficulty question after a positive session', () => {
    const question = selectProfileQuestion(
      input({
        interactions: [
          {
            id: 'training-1',
            occurredAt: '2026-08-25T19:30:00.000Z',
            kind: 'COMPLETED',
            questFamily: 'TRAINING',
            dogExperience: 'comfortable',
          },
        ],
      })
    );

    expect(question?.target).toEqual({ kind: 'SESSION_DIFFICULTY', questFamily: 'TRAINING' });
  });

  it('lets repeated positive social outcomes suppress a redundant social-comfort question', () => {
    const socialOutcomes = [1, 2, 3, 4].map((index) => ({
      id: `social-${index}`,
      occurredAt: `2026-08-${20 + index}T18:00:00.000Z`,
      kind: 'COMPLETED' as const,
      questFamily: 'SOCIAL' as const,
      dogExperience: 'comfortable' as const,
    }));

    expect(
      selectProfileQuestion(
        input({
          profileEvidence: allKnown(['DOG_SOCIAL_COMFORT']),
          interactions: socialOutcomes,
        })
      )
    ).toBeNull();
  });

  it('does not let one replayed positive interaction inflate social confidence', () => {
    const socialOutcome = {
      id: 'social-1',
      occurredAt: '2026-08-25T18:00:00.000Z',
      kind: 'COMPLETED' as const,
      questFamily: 'SOCIAL' as const,
      dogExperience: 'comfortable' as const,
    };
    const question = selectProfileQuestion(
      input({
        profileEvidence: allKnown(['DOG_SOCIAL_COMFORT']),
        interactions: [socialOutcome, socialOutcome, socialOutcome, socialOutcome],
      })
    );

    expect(question?.target).toEqual({ kind: 'PROFILE', dimension: 'DOG_SOCIAL_COMFORT' });
  });

  it('keeps an explicit owner correction authoritative over inferred positive history', () => {
    const socialOutcomes = [1, 2, 3, 4].map((index) => ({
      id: `social-${index}`,
      occurredAt: `2026-08-${20 + index}T18:00:00.000Z`,
      kind: 'COMPLETED' as const,
      questFamily: 'SOCIAL' as const,
      dogExperience: 'comfortable' as const,
    }));

    const question = selectProfileQuestion(
      input({
        profileEvidence: [
          ...allKnown(['DOG_SOCIAL_COMFORT']),
          evidence('DOG_SOCIAL_COMFORT', {
            state: 'LEARNING',
            confidence: 0.4,
            provenance: 'OWNER_CORRECTION',
          }),
        ],
        interactions: socialOutcomes,
      })
    );

    expect(question?.target).toEqual({ kind: 'PROFILE', dimension: 'DOG_SOCIAL_COMFORT' });
    expect(question?.reasonCodes).toContain('OWNER_CORRECTION_UNCERTAIN');
  });

  it('ignores future and malformed profile evidence instead of letting it suppress a question', () => {
    const question = selectProfileQuestion(
      input({
        profileEvidence: [
          evidence('OWNER_GOALS', { updatedAt: '2026-08-26T20:00:00.000Z' }),
          evidence('OWNER_GOALS', { confidence: Number.NaN }),
        ],
      })
    );

    expect(question?.id).toBe('profile-owner-goals-v1');
  });

  it('uses deterministic profile-state authority when otherwise identical evidence conflicts', () => {
    const question = selectProfileQuestion(
      input({
        profileEvidence: [
          evidence('OWNER_GOALS', { state: 'UNKNOWN', confidence: 0.95 }),
          evidence('OWNER_GOALS', { state: 'KNOWN', confidence: 0.95 }),
          ...allKnown(['OWNER_GOALS']),
        ],
      })
    );

    expect(question).toBeNull();
  });

  it('is byte-stable under input permutation', () => {
    const profileEvidence = [
      evidence('OWNER_GOALS', { confidence: 0.6, state: 'LEARNING' }),
      evidence('OWNER_TIME_BUDGET'),
    ];
    const interactions = [
      {
        id: 'dismiss-b',
        occurredAt: '2026-08-25T18:00:00.000Z',
        kind: 'DISMISSED' as const,
        questFamily: 'SCENT' as const,
      },
      {
        id: 'dismiss-a',
        occurredAt: '2026-08-24T18:00:00.000Z',
        kind: 'DISMISSED' as const,
        questFamily: 'SCENT' as const,
      },
    ];

    const first = selectProfileQuestion(input({ profileEvidence, interactions }));
    const second = selectProfileQuestion(
      input({
        profileEvidence: [...profileEvidence].reverse(),
        interactions: [...interactions].reverse(),
      })
    );

    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  });

  it('requires an explicit valid clock', () => {
    expect(() => selectProfileQuestion(input({ now: 'not-a-time' }))).toThrow(
      'profile-question-policy-v1 requires a valid explicit now timestamp'
    );
  });

  it('does not expose game currency as a model/question-policy target', () => {
    const serialized = JSON.stringify(selectProfileQuestion(input()));

    expect(serialized).not.toMatch(/\b(?:xp|reward)\b/i);
  });
});
