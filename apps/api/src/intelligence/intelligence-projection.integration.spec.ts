import { randomUUID } from 'node:crypto';
import { Prisma } from '@woof/database';
import { HouseholdsService } from '../households/households.service';
import { PrismaService } from '../prisma/prisma.service';
import {
  normalizeActivityMeasurement,
  normalizeOwnerCheckinObservation,
} from './evidence-normalization-v1';
import { IntelligenceProjectionService } from './intelligence-projection.service';
import type { ProjectionObservationCandidate } from './evidence-projection-v1.types';

type Fixture = {
  ownerId: string;
  petId: string;
};

const NOW = '2026-08-25T23:00:00.000Z';
const OBSERVED_AT = '2026-08-25T18:00:00.000Z';
const LOCAL_DATE = '2026-08-25';

describe('IntelligenceProjectionService integration', () => {
  const prisma = new PrismaService();
  const households = new HouseholdsService(prisma);
  const service = new IntelligenceProjectionService(prisma, households);
  const usersToDelete: string[] = [];

  beforeAll(async () => {
    await prisma.$connect();
  });

  afterAll(async () => {
    if (usersToDelete.length > 0) {
      await prisma.user.deleteMany({ where: { id: { in: usersToDelete } } });
    }
    await prisma.$disconnect();
  });

  async function createUser(label: string) {
    const suffix = randomUUID().slice(0, 8);
    const user = await prisma.user.create({
      data: {
        handle: `intel-${label}-${suffix}`,
        email: `intel-${label}-${suffix}@example.test`,
      },
      select: { id: true },
    });
    usersToDelete.push(user.id);
    return user;
  }

  async function fixture(label: string): Promise<Fixture> {
    const owner = await createUser(`${label}-owner`);
    const pet = await prisma.pet.create({
      data: {
        ownerId: owner.id,
        name: `Intel ${label}`,
        species: 'DOG',
      },
      select: { id: true },
    });
    return { ownerId: owner.id, petId: pet.id };
  }

  function ownerCandidate(input: {
    ownerId: string;
    petId: string;
    careEventId?: string;
    choice?: 'LESS' | 'USUAL' | 'MORE';
    dimension?: 'APPETITE' | 'ENERGY';
    supersedesObservationId?: string;
    observedAt?: string;
  }): ProjectionObservationCandidate {
    const candidate = normalizeOwnerCheckinObservation({
      userId: input.ownerId,
      petId: input.petId,
      careEventId: input.careEventId ?? randomUUID(),
      dimension: input.dimension ?? 'APPETITE',
      choice: input.choice ?? 'USUAL',
      observedAt: input.observedAt ?? OBSERVED_AT,
      localDate: LOCAL_DATE,
      ...(input.supersedesObservationId
        ? { supersedesObservationId: input.supersedesObservationId }
        : {}),
    });
    if (!candidate) throw new Error('fixture owner candidate unexpectedly normalized to null');
    return candidate;
  }

  it('makes 20 simultaneous identical projection writes one logical observation', async () => {
    const { ownerId, petId } = await fixture('replay-race');
    const candidate = ownerCandidate({ ownerId, petId, careEventId: randomUUID() });

    const receipts = await Promise.all(
      Array.from({ length: 20 }, () => service.projectObservation(candidate))
    );

    expect(receipts.filter((receipt) => !receipt.duplicate)).toHaveLength(1);
    expect(receipts.filter((receipt) => receipt.duplicate)).toHaveLength(19);
    expect(new Set(receipts.map((receipt) => receipt.observationId)).size).toBe(1);
    expect(new Set(receipts.map((receipt) => receipt.payloadHash)).size).toBe(1);

    const rows = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM dogos_intelligence.observations
      WHERE pet_id = ${petId}
        AND source_identity = ${candidate.sourceIdentity}
    `);
    expect(rows[0]?.count).toBe(1);
  });

  it('rejects the same source identity replayed with different semantics', async () => {
    const { ownerId, petId } = await fixture('divergent-replay');
    const candidate = ownerCandidate({ ownerId, petId, careEventId: randomUUID() });
    await service.projectObservation(candidate);

    await expect(
      service.projectObservation({
        ...candidate,
        deltaBucket: candidate.deltaBucket === 0 ? 1 : 0,
      })
    ).rejects.toThrow('different semantics');
  });

  it('serializes competing corrections so exactly one active successor wins', async () => {
    const { ownerId, petId } = await fixture('correction-race');
    const original = await service.projectObservation(
      ownerCandidate({ ownerId, petId, careEventId: randomUUID(), choice: 'LESS' })
    );

    const correctionA = ownerCandidate({
      ownerId,
      petId,
      careEventId: randomUUID(),
      choice: 'USUAL',
      supersedesObservationId: original.observationId,
    });
    const correctionB = ownerCandidate({
      ownerId,
      petId,
      careEventId: randomUUID(),
      choice: 'MORE',
      supersedesObservationId: original.observationId,
    });

    const settled = await Promise.allSettled([
      service.projectObservation(correctionA),
      service.projectObservation(correctionB),
    ]);
    const fulfilled = settled.filter(
      (
        result
      ): result is PromiseFulfilledResult<Awaited<ReturnType<typeof service.projectObservation>>> =>
        result.status === 'fulfilled'
    );
    const rejected = settled.filter((result) => result.status === 'rejected');

    expect(fulfilled).toHaveLength(1);
    expect(rejected).toHaveLength(1);

    const effective = await service.getBaselineEvidence({
      userId: ownerId,
      petId,
      dimension: 'APPETITE',
      now: NOW,
    });
    expect(effective).toHaveLength(1);
    expect(effective[0]?.id).toBe(fulfilled[0]?.value.observationId);
    expect(effective[0]?.id).not.toBe(original.observationId);

    const activeSuccessors = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM dogos_intelligence.observations
      WHERE supersedes_observation_id = ${original.observationId}
        AND retracted_at IS NULL
    `);
    expect(activeSuccessors[0]?.count).toBe(1);
  });

  it('retraction never resurrects superseded evidence without an explicit replacement', async () => {
    const { ownerId, petId } = await fixture('correction-retraction');
    const original = await service.projectObservation(
      ownerCandidate({ ownerId, petId, careEventId: randomUUID(), choice: 'LESS' })
    );
    const correction = await service.projectObservation(
      ownerCandidate({
        ownerId,
        petId,
        careEventId: randomUUID(),
        choice: 'USUAL',
        supersedesObservationId: original.observationId,
      })
    );

    const before = await service.getBaselineEvidence({
      userId: ownerId,
      petId,
      dimension: 'APPETITE',
      now: NOW,
    });
    expect(before.map((row) => row.id)).toEqual([correction.observationId]);

    const retraction = await service.retractObservation({
      userId: ownerId,
      petId,
      observationId: correction.observationId,
      reason: 'Correction withdrawn by authorized household member',
    });
    expect(retraction).toEqual({
      observationId: correction.observationId,
      retracted: true,
      duplicate: false,
    });
    expect(
      await service.retractObservation({
        userId: ownerId,
        petId,
        observationId: correction.observationId,
        reason: 'Correction withdrawn by authorized household member',
      })
    ).toEqual({
      observationId: correction.observationId,
      retracted: true,
      duplicate: true,
    });

    const afterRetraction = await service.getBaselineEvidence({
      userId: ownerId,
      petId,
      dimension: 'APPETITE',
      now: NOW,
    });
    expect(afterRetraction).toEqual([]);

    const explicitReplacement = await service.projectObservation(
      ownerCandidate({
        ownerId,
        petId,
        careEventId: randomUUID(),
        choice: 'LESS',
        supersedesObservationId: original.observationId,
      })
    );
    const afterReplacement = await service.getBaselineEvidence({
      userId: ownerId,
      petId,
      dimension: 'APPETITE',
      now: NOW,
    });
    expect(afterReplacement.map((row) => row.id)).toEqual([explicitReplacement.observationId]);
    expect(afterReplacement.map((row) => row.id)).not.toContain(original.observationId);

    const historicalRows = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM dogos_intelligence.observations
      WHERE id IN (
        ${original.observationId},
        ${correction.observationId},
        ${explicitReplacement.observationId}
      )
    `);
    expect(historicalRows[0]?.count).toBe(3);
  });

  it('keeps Activity measurements context-only and outside baseline evidence', async () => {
    const { ownerId, petId } = await fixture('activity-context');
    const owner = await service.projectObservation(
      ownerCandidate({ ownerId, petId, careEventId: randomUUID(), choice: 'USUAL' })
    );
    const activity = await service.projectObservation(
      normalizeActivityMeasurement({
        userId: ownerId,
        petId,
        activityId: randomUUID(),
        dimension: 'ACTIVITY_LOAD',
        numericValue: 45,
        unit: 'minutes',
        observedAt: OBSERVED_AT,
        localDate: LOCAL_DATE,
      })
    );

    const baseline = await service.getBaselineEvidence({
      userId: ownerId,
      petId,
      dimension: 'APPETITE',
      now: NOW,
    });
    expect(baseline.map((row) => row.id)).toEqual([owner.observationId]);
    expect(baseline.map((row) => row.id)).not.toContain(activity.observationId);

    const activityHistory = await service.getEffectiveProjectionHistory({
      userId: ownerId,
      petId,
      dimension: 'ACTIVITY_LOAD',
      from: '2026-08-25T00:00:00.000Z',
      to: NOW,
    });
    expect(activityHistory).toHaveLength(1);
    expect(activityHistory[0]).toMatchObject({
      id: activity.observationId,
      authority: 'CONTEXT_ONLY',
      numericValue: 45,
      unit: 'minutes',
    });
  });

  it('enforces the same household pet authority on projection writes and reads', async () => {
    const { ownerId, petId } = await fixture('household-authority');
    const member = await createUser('projection-member');
    const outsider = await createUser('projection-outsider');
    const householdId = await households.ensurePersonalHousehold(ownerId);

    await prisma.householdMember.upsert({
      where: { householdId_userId: { householdId, userId: member.id } },
      update: { status: 'ACTIVE', role: 'MEMBER' },
      create: { householdId, userId: member.id, status: 'ACTIVE', role: 'MEMBER' },
    });

    const memberCandidate = ownerCandidate({
      ownerId: member.id,
      petId,
      careEventId: randomUUID(),
      choice: 'USUAL',
    });
    const memberReceipt = await service.projectObservation(memberCandidate);
    expect(memberReceipt.duplicate).toBe(false);

    const memberRead = await service.getBaselineEvidence({
      userId: member.id,
      petId,
      dimension: 'APPETITE',
      now: NOW,
    });
    expect(memberRead.map((row) => row.id)).toEqual([memberReceipt.observationId]);

    const outsiderCandidate = ownerCandidate({
      ownerId: outsider.id,
      petId,
      careEventId: randomUUID(),
      choice: 'MORE',
    });
    await expect(service.projectObservation(outsiderCandidate)).rejects.toThrow('Pet not found');
    await expect(
      service.getBaselineEvidence({
        userId: outsider.id,
        petId,
        dimension: 'APPETITE',
        now: NOW,
      })
    ).rejects.toThrow('Pet not found');
  });

  it('returns effective evidence in deterministic observed-time then canonical-id order', async () => {
    const { ownerId, petId } = await fixture('ordering');
    const later = await service.projectObservation(
      ownerCandidate({
        ownerId,
        petId,
        careEventId: randomUUID(),
        dimension: 'ENERGY',
        observedAt: '2026-08-25T20:00:00.000Z',
      })
    );
    const earlier = await service.projectObservation(
      ownerCandidate({
        ownerId,
        petId,
        careEventId: randomUUID(),
        dimension: 'ENERGY',
        observedAt: '2026-08-25T16:00:00.000Z',
      })
    );

    const evidence = await service.getBaselineEvidence({
      userId: ownerId,
      petId,
      dimension: 'ENERGY',
      now: NOW,
    });
    expect(evidence.map((row) => row.id)).toEqual([earlier.observationId, later.observationId]);
  });
});
