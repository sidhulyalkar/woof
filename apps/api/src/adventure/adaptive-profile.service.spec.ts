import { BadRequestException, ConflictException } from '@nestjs/common';
import type { HouseholdsService } from '../households/households.service';
import type { PrismaService } from '../prisma/prisma.service';
import { ADAPTIVE_PROFILE_SCHEMA_VERSION } from './adaptive-profile-v1';
import { AdaptiveProfileService } from './adaptive-profile.service';

const HOUSEHOLD_ID = 'household-1';
const PET_ID = 'pet-1';
const USER_ID = 'user-1';

type StoredEvidence = {
  id: string;
  householdId: string;
  petId: string;
  dimension: string;
  subject: string;
  state: string;
  value: unknown;
  confidence: number;
  provenance: string;
  schemaVersion: string;
  sourceUserId: string | null;
  occurredAt: Date;
  createdAt: Date;
};

type StoredQuestion = {
  id: string;
  householdId: string;
  petId: string;
  userId: string;
  questionId: string;
  policyVersion: string;
  outcome: string;
  answer: unknown;
  askedAt: Date;
  respondedAt: Date;
  createdAt: Date;
};

function makeHarness() {
  const evidence: StoredEvidence[] = [];
  const questions: StoredQuestion[] = [];

  const adaptiveProfileEvidence = {
    findUnique: jest.fn(async ({ where }: { where: { id: string } }) =>
      evidence.find((entry) => entry.id === where.id)
    ),
    findMany: jest.fn(
      async ({ where }: { where: { householdId: string; petId: string; schemaVersion: string } }) =>
        evidence.filter(
          (entry) =>
            entry.householdId === where.householdId &&
            entry.petId === where.petId &&
            entry.schemaVersion === where.schemaVersion
        )
    ),
    create: jest.fn(async ({ data }: { data: Record<string, unknown> }) => {
      const row: StoredEvidence = {
        id: String(data.id),
        householdId: String(data.householdId),
        petId: String(data.petId),
        dimension: String(data.dimension),
        subject: String(data.subject),
        state: String(data.state),
        value: data.value === undefined ? null : data.value,
        confidence: Number(data.confidence),
        provenance: String(data.provenance),
        schemaVersion: String(data.schemaVersion),
        sourceUserId: data.sourceUserId === null ? null : String(data.sourceUserId),
        occurredAt: data.occurredAt as Date,
        createdAt: new Date(),
      };
      if (evidence.some((entry) => entry.id === row.id)) throw new Error('duplicate evidence');
      evidence.push(row);
      return row;
    }),
  };

  const adaptiveProfileQuestionResponse = {
    findUnique: jest.fn(async ({ where }: { where: { id: string } }) =>
      questions.find((entry) => entry.id === where.id)
    ),
    findMany: jest.fn(async ({ where }: { where: { householdId: string; petId: string } }) =>
      questions
        .filter((entry) => entry.householdId === where.householdId && entry.petId === where.petId)
        .sort((left, right) => right.askedAt.getTime() - left.askedAt.getTime())
    ),
    create: jest.fn(async ({ data }: { data: Record<string, unknown> }) => {
      const row: StoredQuestion = {
        id: String(data.id),
        householdId: String(data.householdId),
        petId: String(data.petId),
        userId: String(data.userId),
        questionId: String(data.questionId),
        policyVersion: String(data.policyVersion),
        outcome: String(data.outcome),
        answer: data.answer === undefined ? null : data.answer,
        askedAt: data.askedAt as Date,
        respondedAt: data.respondedAt as Date,
        createdAt: new Date(),
      };
      if (questions.some((entry) => entry.id === row.id)) throw new Error('duplicate question');
      questions.push(row);
      return row;
    }),
  };

  const tx = { adaptiveProfileEvidence, adaptiveProfileQuestionResponse };
  const prisma = {
    adaptiveProfileEvidence,
    adaptiveProfileQuestionResponse,
    $transaction: jest.fn(async (callback: (client: typeof tx) => Promise<unknown>) =>
      callback(tx)
    ),
  };
  const households = {
    assertHouseholdPetAccessible: jest.fn(async () => ({
      householdId: HOUSEHOLD_ID,
      petId: PET_ID,
      timezone: 'UTC',
    })),
  };

  return {
    evidence,
    questions,
    prisma,
    households,
    service: new AdaptiveProfileService(
      prisma as unknown as PrismaService,
      households as unknown as HouseholdsService
    ),
  };
}

describe('AdaptiveProfileService', () => {
  it('authorizes and reads only the requested household/pet pair', async () => {
    const harness = makeHarness();
    harness.evidence.push({
      id: 'other-pet',
      householdId: HOUSEHOLD_ID,
      petId: 'pet-2',
      dimension: 'DOG_ENERGY_PATTERN',
      subject: 'DOG',
      state: 'KNOWN',
      value: ['OFTEN_ACTIVE'],
      confidence: 1,
      provenance: 'OWNER_EXPLICIT',
      schemaVersion: ADAPTIVE_PROFILE_SCHEMA_VERSION,
      sourceUserId: USER_ID,
      occurredAt: new Date(),
      createdAt: new Date(),
    });

    const result = await harness.service.getState(USER_ID, HOUSEHOLD_ID, PET_ID);

    expect(harness.households.assertHouseholdPetAccessible).toHaveBeenCalledWith(
      USER_ID,
      HOUSEHOLD_ID,
      PET_ID
    );
    expect(harness.prisma.adaptiveProfileEvidence.findMany).toHaveBeenCalledWith(
      expect.objectContaining({
        where: {
          householdId: HOUSEHOLD_ID,
          petId: PET_ID,
          schemaVersion: ADAPTIVE_PROFILE_SCHEMA_VERSION,
        },
      })
    );
    expect(result.coverage.known).toEqual([]);
    expect(result.coverage.unknown).toContain('DOG_ENERGY_PATTERN');
  });

  it('records a static answer and its durable evidence atomically', async () => {
    const harness = makeHarness();

    const result = await harness.service.recordQuestionResponse(USER_ID, HOUSEHOLD_ID, PET_ID, {
      responseId: 'response-1',
      questionId: 'profile-owner-goals-v1',
      outcome: 'ANSWERED',
      answers: ['TRAINING', 'MORE_ADVENTURES'],
    });

    expect(result.duplicate).toBe(false);
    expect(harness.prisma.$transaction).toHaveBeenCalledTimes(1);
    expect(harness.questions).toHaveLength(1);
    expect(harness.evidence).toHaveLength(1);
    expect(harness.evidence[0]).toEqual(
      expect.objectContaining({
        householdId: HOUSEHOLD_ID,
        petId: PET_ID,
        dimension: 'OWNER_GOALS',
        subject: 'OWNER',
        state: 'KNOWN',
        confidence: 1,
        provenance: 'OWNER_EXPLICIT',
        schemaVersion: ADAPTIVE_PROFILE_SCHEMA_VERSION,
        sourceUserId: USER_ID,
      })
    );
    expect(result.profile.coverage.known).toContain('OWNER_GOALS');
  });

  it('records skip as question history only', async () => {
    const harness = makeHarness();

    await harness.service.recordQuestionResponse(USER_ID, HOUSEHOLD_ID, PET_ID, {
      responseId: 'skip-1',
      questionId: 'profile-dog-energy-v1',
      outcome: 'SKIPPED',
    });

    expect(harness.questions).toHaveLength(1);
    expect(harness.evidence).toHaveLength(0);
  });

  it('records not-sure as explicit unknown rather than a negative preference', async () => {
    const harness = makeHarness();

    const result = await harness.service.recordQuestionResponse(USER_ID, HOUSEHOLD_ID, PET_ID, {
      responseId: 'not-sure-1',
      questionId: 'profile-dog-social-comfort-v1',
      outcome: 'NOT_SURE',
    });

    expect(harness.evidence[0]).toEqual(
      expect.objectContaining({
        dimension: 'DOG_SOCIAL_COMFORT',
        state: 'UNKNOWN',
        value: expect.anything(),
        confidence: 0,
        provenance: 'OWNER_EXPLICIT',
      })
    );
    const dimension = result.profile.dimensions.find(
      (entry) => entry.dimension === 'DOG_SOCIAL_COMFORT'
    );
    expect(dimension?.state).toBe('UNKNOWN');
    expect(dimension?.confidence).toBe(0);
  });

  it('logs dynamic micro-question history without inventing durable profile evidence', async () => {
    const harness = makeHarness();

    await harness.service.recordQuestionResponse(USER_ID, HOUSEHOLD_ID, PET_ID, {
      responseId: 'dynamic-1',
      questionId: 'quest-difficulty:training-v1',
      outcome: 'ANSWERED',
      answers: ['ABOUT_RIGHT'],
    });

    expect(harness.questions).toHaveLength(1);
    expect(harness.evidence).toHaveLength(0);
  });

  it('deduplicates exact response replays and fails closed on divergent identity reuse', async () => {
    const harness = makeHarness();
    const original = {
      responseId: 'replay-1',
      questionId: 'profile-owner-time-budget-v1',
      outcome: 'ANSWERED' as const,
      answers: ['FIVE_MIN'],
    };

    await harness.service.recordQuestionResponse(USER_ID, HOUSEHOLD_ID, PET_ID, original);
    const replay = await harness.service.recordQuestionResponse(
      USER_ID,
      HOUSEHOLD_ID,
      PET_ID,
      original
    );

    expect(replay.duplicate).toBe(true);
    expect(harness.questions).toHaveLength(1);
    expect(harness.evidence).toHaveLength(1);

    await expect(
      harness.service.recordQuestionResponse(USER_ID, HOUSEHOLD_ID, PET_ID, {
        ...original,
        answers: ['FORTY_PLUS'],
      })
    ).rejects.toBeInstanceOf(ConflictException);
  });

  it('makes corrections replay-safe and owner-authoritative', async () => {
    const harness = makeHarness();
    const correction = {
      mutationId: 'correction-1',
      dimension: 'DOG_NOVELTY_COMFORT' as const,
      state: 'KNOWN' as const,
      values: ['PREFERS_FAMILIAR'],
    };

    const first = await harness.service.correct(USER_ID, HOUSEHOLD_ID, PET_ID, correction);
    const replay = await harness.service.correct(USER_ID, HOUSEHOLD_ID, PET_ID, correction);

    expect(first.duplicate).toBe(false);
    expect(replay.duplicate).toBe(true);
    expect(harness.evidence).toHaveLength(1);
    expect(harness.evidence[0]?.provenance).toBe('OWNER_CORRECTION');

    await expect(
      harness.service.correct(USER_ID, HOUSEHOLD_ID, PET_ID, {
        ...correction,
        values: ['USUALLY_CURIOUS'],
      })
    ).rejects.toBeInstanceOf(ConflictException);
  });

  it('rejects malformed static answers as a client error', async () => {
    const harness = makeHarness();

    await expect(
      harness.service.recordQuestionResponse(USER_ID, HOUSEHOLD_ID, PET_ID, {
        responseId: 'bad-answer',
        questionId: 'profile-dog-energy-v1',
        outcome: 'ANSWERED',
        answers: ['IMPOSSIBLE_ENUM'],
      })
    ).rejects.toBeInstanceOf(BadRequestException);
  });

  it('does not touch persistence when household authorization fails', async () => {
    const harness = makeHarness();
    harness.households.assertHouseholdPetAccessible.mockRejectedValueOnce(
      new Error('not authorized')
    );

    await expect(harness.service.getState(USER_ID, HOUSEHOLD_ID, PET_ID)).rejects.toThrow(
      'not authorized'
    );
    expect(harness.prisma.adaptiveProfileEvidence.findMany).not.toHaveBeenCalled();
  });
});
