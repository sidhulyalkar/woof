import { BadRequestException, ConflictException, Injectable } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { HouseholdsService } from '../households/households.service';
import { PrismaService } from '../prisma/prisma.service';
import type {
  CorrectAdaptiveProfileDto,
  RecordProfileQuestionResponseDto,
} from './dto/adaptive-profile.dto';
import {
  ADAPTIVE_PROFILE_SCHEMA_VERSION,
  normalizeQuestionAnswer,
  profileQuestionPolicyVersion,
  profileSubjectForDimension,
  resolveCurrentProfileEvidence,
  toPolicyEvidence,
  type PersistedProfileEvidenceLike,
} from './adaptive-profile-v1';
import {
  PROFILE_DIMENSIONS,
  type ProfileDimension,
  type QuestionHistoryEntry,
} from './profile-question-policy-v1.types';

type EvidenceRow = PersistedProfileEvidenceLike & {
  schemaVersion: string;
};

type QuestionResponseRow = {
  id: string;
  householdId: string;
  petId: string;
  userId: string;
  questionId: string;
  policyVersion: string;
  outcome: string;
  answer: Prisma.JsonValue | null;
  askedAt: Date;
  respondedAt: Date;
};

type ProfileDimensionState = {
  dimension: ProfileDimension;
  subject: 'DOG' | 'OWNER' | 'PAIR';
  state: 'UNKNOWN' | 'LEARNING' | 'KNOWN';
  value: unknown;
  confidence: number;
  provenance: string | null;
  updatedAt: string | null;
};

export type AdaptiveProfileState = {
  schemaVersion: typeof ADAPTIVE_PROFILE_SCHEMA_VERSION;
  householdId: string;
  petId: string;
  dimensions: ProfileDimensionState[];
  coverage: {
    known: ProfileDimension[];
    learning: ProfileDimension[];
    unknown: ProfileDimension[];
  };
};

export type AdaptiveProfilePolicySnapshot = {
  profile: NonNullable<ReturnType<typeof toPolicyEvidence>>[];
  questionHistory: QuestionHistoryEntry[];
};

function normalizeBoundedValues(values: readonly string[] | undefined): string[] {
  const normalized = [
    ...new Set((values ?? []).map((value) => value.trim()).filter(Boolean)),
  ].sort();
  if (normalized.length > 8 || normalized.some((value) => value.length > 80)) {
    throw new BadRequestException('Adaptive profile values exceed the bounded payload contract');
  }
  return normalized;
}

function jsonMatches(value: Prisma.JsonValue | null, expected: readonly string[] | null): boolean {
  if (expected === null) return value === null;
  if (!Array.isArray(value)) return false;
  return JSON.stringify([...value].sort()) === JSON.stringify([...expected].sort());
}

function isQuestionOutcome(value: string): value is QuestionHistoryEntry['outcome'] {
  return value === 'ANSWERED' || value === 'NOT_SURE' || value === 'SKIPPED';
}

@Injectable()
export class AdaptiveProfileService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly households: HouseholdsService
  ) {}

  async getState(
    userId: string,
    householdId: string,
    petId: string
  ): Promise<AdaptiveProfileState> {
    await this.assertAccessible(userId, householdId, petId);
    return this.readState(householdId, petId);
  }

  async getPolicySnapshot(
    userId: string,
    householdId: string,
    petId: string
  ): Promise<AdaptiveProfilePolicySnapshot> {
    await this.assertAccessible(userId, householdId, petId);
    const [evidence, responses] = await Promise.all([
      this.readEvidence(householdId, petId),
      this.prisma.adaptiveProfileQuestionResponse.findMany({
        where: { householdId, petId },
        orderBy: [{ askedAt: 'desc' }, { id: 'desc' }],
        take: 200,
        select: { questionId: true, outcome: true, askedAt: true },
      }),
    ]);

    return {
      profile: resolveCurrentProfileEvidence(evidence).flatMap((entry) => {
        const projected = toPolicyEvidence(entry);
        return projected ? [projected] : [];
      }),
      questionHistory: responses.flatMap((response) =>
        isQuestionOutcome(response.outcome)
          ? [
              {
                questionId: response.questionId,
                outcome: response.outcome,
                askedAt: response.askedAt.toISOString(),
              },
            ]
          : []
      ),
    };
  }

  async recordQuestionResponse(
    userId: string,
    householdId: string,
    petId: string,
    dto: RecordProfileQuestionResponseDto
  ): Promise<{ duplicate: boolean; profile: AdaptiveProfileState }> {
    await this.assertAccessible(userId, householdId, petId);

    const responseId = `profile-question:${dto.responseId}`;
    const evidenceId = `profile-evidence:question:${dto.responseId}`;
    const now = new Date();
    const policyVersion = profileQuestionPolicyVersion();

    if (dto.outcome !== 'ANSWERED' && dto.answers?.length) {
      throw new BadRequestException('Skip and not-sure responses cannot carry answers');
    }

    let staticAnswer: ReturnType<typeof normalizeQuestionAnswer>;
    try {
      staticAnswer = normalizeQuestionAnswer(dto.questionId, dto.outcome, dto.answers);
    } catch (error) {
      throw new BadRequestException(
        error instanceof Error ? error.message : 'Profile answer is invalid for this question'
      );
    }
    const normalizedAnswers = staticAnswer
      ? staticAnswer.values
      : dto.outcome === 'ANSWERED'
        ? normalizeBoundedValues(dto.answers)
        : null;

    if (dto.outcome === 'ANSWERED' && !normalizedAnswers?.length) {
      throw new BadRequestException('Answered profile questions require at least one answer');
    }

    const execute = async () =>
      this.prisma.$transaction(async (tx) => {
        const existing = await tx.adaptiveProfileQuestionResponse.findUnique({
          where: { id: responseId },
        });
        if (existing) {
          this.assertQuestionReplayMatches(existing, {
            householdId,
            petId,
            userId,
            questionId: dto.questionId,
            policyVersion,
            outcome: dto.outcome,
            answers: normalizedAnswers,
          });
          return true;
        }

        await tx.adaptiveProfileQuestionResponse.create({
          data: {
            id: responseId,
            householdId,
            petId,
            userId,
            questionId: dto.questionId,
            policyVersion,
            outcome: dto.outcome,
            answer: normalizedAnswers === null ? Prisma.JsonNull : normalizedAnswers,
            askedAt: now,
            respondedAt: now,
          },
        });

        if (staticAnswer && dto.outcome !== 'SKIPPED') {
          await tx.adaptiveProfileEvidence.create({
            data: {
              id: evidenceId,
              householdId,
              petId,
              dimension: staticAnswer.dimension,
              subject: profileSubjectForDimension(staticAnswer.dimension),
              state: dto.outcome === 'ANSWERED' ? 'KNOWN' : 'UNKNOWN',
              value: normalizedAnswers === null ? Prisma.JsonNull : normalizedAnswers,
              confidence: dto.outcome === 'ANSWERED' ? 1 : 0,
              provenance: 'OWNER_EXPLICIT',
              schemaVersion: ADAPTIVE_PROFILE_SCHEMA_VERSION,
              sourceUserId: userId,
              occurredAt: now,
            },
          });
        }
        return false;
      });

    let duplicate: boolean;
    try {
      duplicate = await execute();
    } catch (error) {
      if (!this.isUniqueViolation(error)) throw error;
      const existing = await this.prisma.adaptiveProfileQuestionResponse.findUnique({
        where: { id: responseId },
      });
      if (!existing) throw error;
      this.assertQuestionReplayMatches(existing, {
        householdId,
        petId,
        userId,
        questionId: dto.questionId,
        policyVersion,
        outcome: dto.outcome,
        answers: normalizedAnswers,
      });
      duplicate = true;
    }

    return { duplicate, profile: await this.readState(householdId, petId) };
  }

  async correct(
    userId: string,
    householdId: string,
    petId: string,
    dto: CorrectAdaptiveProfileDto
  ): Promise<{ duplicate: boolean; profile: AdaptiveProfileState }> {
    await this.assertAccessible(userId, householdId, petId);

    const values = normalizeBoundedValues(dto.values);
    if (dto.state === 'KNOWN' && values.length === 0) {
      throw new BadRequestException('Known profile corrections require a value');
    }
    if (dto.state === 'UNKNOWN' && values.length > 0) {
      throw new BadRequestException('Unknown profile corrections cannot carry values');
    }
    if (values.some((value) => value === 'SKIP' || value === 'NOT_SURE')) {
      throw new BadRequestException(
        'Use explicit UNKNOWN instead of skip/not-sure correction values'
      );
    }

    const evidenceId = `profile-evidence:correction:${dto.mutationId}`;
    const now = new Date();
    const expectedValues = dto.state === 'KNOWN' ? values : null;

    const execute = async () =>
      this.prisma.$transaction(async (tx) => {
        const existing = await tx.adaptiveProfileEvidence.findUnique({ where: { id: evidenceId } });
        if (existing) {
          this.assertCorrectionReplayMatches(existing, {
            householdId,
            petId,
            userId,
            dimension: dto.dimension,
            state: dto.state,
            values: expectedValues,
          });
          return true;
        }

        await tx.adaptiveProfileEvidence.create({
          data: {
            id: evidenceId,
            householdId,
            petId,
            dimension: dto.dimension,
            subject: profileSubjectForDimension(dto.dimension),
            state: dto.state,
            value: expectedValues === null ? Prisma.JsonNull : expectedValues,
            confidence: dto.state === 'KNOWN' ? 1 : 0,
            provenance: 'OWNER_CORRECTION',
            schemaVersion: ADAPTIVE_PROFILE_SCHEMA_VERSION,
            sourceUserId: userId,
            occurredAt: now,
          },
        });
        return false;
      });

    let duplicate: boolean;
    try {
      duplicate = await execute();
    } catch (error) {
      if (!this.isUniqueViolation(error)) throw error;
      const existing = await this.prisma.adaptiveProfileEvidence.findUnique({
        where: { id: evidenceId },
      });
      if (!existing) throw error;
      this.assertCorrectionReplayMatches(existing, {
        householdId,
        petId,
        userId,
        dimension: dto.dimension,
        state: dto.state,
        values: expectedValues,
      });
      duplicate = true;
    }

    return { duplicate, profile: await this.readState(householdId, petId) };
  }

  private async assertAccessible(
    userId: string,
    householdId: string,
    petId: string
  ): Promise<void> {
    await this.households.assertHouseholdPetAccessible(userId, householdId, petId);
  }

  private async readEvidence(householdId: string, petId: string): Promise<EvidenceRow[]> {
    return this.prisma.adaptiveProfileEvidence.findMany({
      where: { householdId, petId, schemaVersion: ADAPTIVE_PROFILE_SCHEMA_VERSION },
      orderBy: [{ occurredAt: 'desc' }, { id: 'asc' }],
      select: {
        id: true,
        dimension: true,
        subject: true,
        state: true,
        value: true,
        confidence: true,
        provenance: true,
        schemaVersion: true,
        occurredAt: true,
      },
    });
  }

  private async readState(householdId: string, petId: string): Promise<AdaptiveProfileState> {
    const evidence = await this.readEvidence(householdId, petId);
    const current = new Map(
      resolveCurrentProfileEvidence(evidence).map((entry) => [
        entry.dimension as ProfileDimension,
        entry,
      ])
    );

    const dimensions = PROFILE_DIMENSIONS.map<ProfileDimensionState>((dimension) => {
      const entry = current.get(dimension);
      if (!entry) {
        return {
          dimension,
          subject: profileSubjectForDimension(dimension),
          state: 'UNKNOWN',
          value: null,
          confidence: 0,
          provenance: null,
          updatedAt: null,
        };
      }
      return {
        dimension,
        subject: profileSubjectForDimension(dimension),
        state: entry.state as ProfileDimensionState['state'],
        value: entry.value,
        confidence: entry.confidence,
        provenance: entry.provenance,
        updatedAt: entry.occurredAt.toISOString(),
      };
    });

    return {
      schemaVersion: ADAPTIVE_PROFILE_SCHEMA_VERSION,
      householdId,
      petId,
      dimensions,
      coverage: {
        known: dimensions
          .filter((entry) => entry.state === 'KNOWN')
          .map((entry) => entry.dimension),
        learning: dimensions
          .filter((entry) => entry.state === 'LEARNING')
          .map((entry) => entry.dimension),
        unknown: dimensions
          .filter((entry) => entry.state === 'UNKNOWN')
          .map((entry) => entry.dimension),
      },
    };
  }

  private assertQuestionReplayMatches(
    existing: QuestionResponseRow,
    expected: {
      householdId: string;
      petId: string;
      userId: string;
      questionId: string;
      policyVersion: string;
      outcome: string;
      answers: readonly string[] | null;
    }
  ): void {
    const matches =
      existing.householdId === expected.householdId &&
      existing.petId === expected.petId &&
      existing.userId === expected.userId &&
      existing.questionId === expected.questionId &&
      existing.policyVersion === expected.policyVersion &&
      existing.outcome === expected.outcome &&
      jsonMatches(existing.answer, expected.answers);
    if (!matches) {
      throw new ConflictException('Profile response identity was replayed with divergent content');
    }
  }

  private assertCorrectionReplayMatches(
    existing: EvidenceRow & { sourceUserId?: string | null },
    expected: {
      householdId: string;
      petId: string;
      userId: string;
      dimension: ProfileDimension;
      state: 'KNOWN' | 'UNKNOWN';
      values: readonly string[] | null;
    }
  ): void {
    const row = existing as EvidenceRow & {
      householdId?: string;
      petId?: string;
      sourceUserId?: string | null;
      value: Prisma.JsonValue | null;
    };
    const matches =
      row.householdId === expected.householdId &&
      row.petId === expected.petId &&
      row.sourceUserId === expected.userId &&
      row.dimension === expected.dimension &&
      row.subject === profileSubjectForDimension(expected.dimension) &&
      row.state === expected.state &&
      row.provenance === 'OWNER_CORRECTION' &&
      row.schemaVersion === ADAPTIVE_PROFILE_SCHEMA_VERSION &&
      jsonMatches(row.value, expected.values);
    if (!matches) {
      throw new ConflictException(
        'Profile correction identity was replayed with divergent content'
      );
    }
  }

  private isUniqueViolation(error: unknown): boolean {
    return error instanceof Prisma.PrismaClientKnownRequestError && error.code === 'P2002';
  }
}
