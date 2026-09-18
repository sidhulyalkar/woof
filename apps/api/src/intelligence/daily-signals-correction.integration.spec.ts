import { randomUUID } from 'node:crypto';
import { Prisma } from '@woof/database';
import { CareEventsService } from '../care-events/care-events.service';
import { HouseholdsService } from '../households/households.service';
import { PrismaService } from '../prisma/prisma.service';
import { DailySignalsCorrectionService } from './daily-signals-correction.service';
import { DailySignalsService } from './daily-signals.service';
import { IntelligenceProjectionService } from './intelligence-projection.service';

type Fixture = {
  ownerId: string;
  memberId: string;
  petId: string;
  householdId: string;
};

describe('DailySignalsCorrectionService integration', () => {
  const prisma = new PrismaService();
  const households = new HouseholdsService(prisma);
  const careEvents = new CareEventsService(prisma, households);
  const projection = new IntelligenceProjectionService(prisma, households);
  const dailySignals = new DailySignalsService(households, careEvents, projection);
  const correction = new DailySignalsCorrectionService(
    prisma,
    households,
    careEvents,
    dailySignals,
    projection
  );
  const usersToDelete: string[] = [];
  const householdsToDelete: string[] = [];
  const now = new Date('2026-09-18T19:00:00.000Z');

  beforeAll(async () => {
    await prisma.$connect();
  });

  afterAll(async () => {
    if (householdsToDelete.length > 0) {
      await prisma.household.deleteMany({ where: { id: { in: householdsToDelete } } });
    }
    if (usersToDelete.length > 0) {
      await prisma.user.deleteMany({ where: { id: { in: usersToDelete } } });
    }
    await prisma.$disconnect();
  });

  async function createUser(label: string) {
    const suffix = randomUUID().slice(0, 8);
    const user = await prisma.user.create({
      data: {
        handle: `corr-${label}-${suffix}`,
        email: `corr-${label}-${suffix}@example.test`,
      },
      select: { id: true },
    });
    usersToDelete.push(user.id);
    return user.id;
  }

  async function fixture(label: string): Promise<Fixture> {
    const ownerId = await createUser(`${label}-owner`);
    const memberId = await createUser(`${label}-member`);
    const pet = await prisma.pet.create({
      data: { ownerId, name: `Correction ${label}`, species: 'DOG' },
      select: { id: true },
    });
    const householdId = await households.ensurePersonalHousehold(ownerId);
    householdsToDelete.push(householdId);
    await prisma.household.update({
      where: { id: householdId },
      data: { timezone: 'America/Los_Angeles' },
    });
    await prisma.householdMember.upsert({
      where: { householdId_userId: { householdId, userId: memberId } },
      update: { status: 'ACTIVE', role: 'MEMBER' },
      create: { householdId, userId: memberId, status: 'ACTIVE', role: 'MEMBER' },
    });
    return { ownerId, memberId, petId: pet.id, householdId };
  }

  async function capture(
    value: Fixture,
    signals: Record<string, 'LESS' | 'USUAL' | 'MORE' | 'UNSURE'>,
    note = 'private author note'
  ) {
    return dailySignals.capture(
      value.ownerId,
      {
        householdId: value.householdId,
        petId: value.petId,
        observedAt: now.toISOString(),
        signals,
        note,
      },
      now
    );
  }

  function input(
    value: Fixture,
    expectedCurrentCareEventId: string,
    signals: Record<string, 'LESS' | 'USUAL' | 'MORE' | 'UNSURE'>
  ) {
    return {
      householdId: value.householdId,
      petId: value.petId,
      expectedCurrentCareEventId,
      signals,
    };
  }

  it('reads only effective structured state and never exposes the private free-form note', async () => {
    const value = await fixture('read');
    const original = await capture(value, { appetite: 'USUAL', energy: 'MORE' }, 'owner-only text');

    const state = await correction.getCurrent(
      value.memberId,
      { householdId: value.householdId, petId: value.petId },
      now
    );

    expect(state).toEqual(
      expect.objectContaining({
        rootCareEventId: original.careEventId,
        currentCareEventId: original.careEventId,
        correctionSequence: 0,
        status: 'ORIGINAL',
        signals: { appetite: 'USUAL', energy: 'MORE' },
      })
    );
    expect(state).not.toHaveProperty('note');
  });

  it('appends a zero-XP correction and makes corrected structured evidence effective', async () => {
    const value = await fixture('replace');
    const original = await capture(value, { appetite: 'USUAL', energy: 'MORE' });

    const receipt = await correction.correct(
      value.ownerId,
      input(value, original.careEventId, {
        appetite: 'LESS',
        sleepRest: 'USUAL',
      }),
      now
    );

    expect(receipt).toEqual(
      expect.objectContaining({
        duplicate: false,
        currentAdvanced: false,
        projectedDimensions: expect.arrayContaining(['APPETITE', 'SLEEP_REST']),
        retractedDimensions: expect.arrayContaining(['ENERGY']),
        state: expect.objectContaining({
          rootCareEventId: original.careEventId,
          currentCareEventId: receipt.correctionCareEventId,
          correctionSequence: 1,
          status: 'CORRECTED',
          signals: { appetite: 'LESS', sleepRest: 'USUAL' },
        }),
      })
    );

    const correctionRows = await prisma.$queryRaw<
      Array<{
        id: string;
        context: Record<string, unknown>;
        outcome: Record<string, unknown>;
        bond_xp: number;
      }>
    >(Prisma.sql`
      SELECT ce.id, ce.context, ce.outcome, COALESCE(rl.bond_xp, 0)::int AS bond_xp
      FROM care_events ce
      LEFT JOIN reward_ledger rl ON rl.care_event_id = ce.id
      WHERE ce.id = ${receipt.correctionCareEventId}
    `);
    expect(correctionRows[0]?.bond_xp).toBe(0);
    expect(correctionRows[0]?.context).toMatchObject({
      rootCareEventId: original.careEventId,
      correctsCareEventId: original.careEventId,
      correctionSequence: 1,
      correctionPolicyVersion: 'daily-signals-correction-v1',
    });
    expect(correctionRows[0]?.outcome).toEqual({
      signals: { appetite: 'LESS', sleepRest: 'USUAL' },
    });
    expect(correctionRows[0]?.outcome).not.toHaveProperty('note');

    const active = await prisma.$queryRaw<
      Array<{ source_event_id: string; dimension: string; delta_bucket: number }>
    >(Prisma.sql`
      SELECT o.source_event_id, o.dimension, o.delta_bucket
      FROM dogos_intelligence.observations o
      WHERE o.pet_id = ${value.petId}
        AND o.source_type = 'OWNER_CHECKIN'
        AND o.retracted_at IS NULL
        AND NOT EXISTS (
          SELECT 1
          FROM dogos_intelligence.observations successor
          WHERE successor.supersedes_observation_id = o.id
        )
      ORDER BY o.dimension
    `);
    expect(active).toEqual([
      {
        source_event_id: receipt.correctionCareEventId,
        dimension: 'APPETITE',
        delta_bucket: -1,
      },
      {
        source_event_id: receipt.correctionCareEventId,
        dimension: 'SLEEP_REST',
        delta_bucket: 0,
      },
    ]);
  });

  it('makes an exact correction retry idempotent and repairs a missing derived projection', async () => {
    const value = await fixture('repair');
    const original = await capture(value, { energy: 'MORE' });
    const request = input(value, original.careEventId, { energy: 'LESS' });

    const first = await correction.correct(value.ownerId, request, now);
    await prisma.$executeRaw(Prisma.sql`
      DELETE FROM dogos_intelligence.observations
      WHERE source_event_id = ${first.correctionCareEventId}
        AND dimension = 'ENERGY'
    `);

    const retry = await correction.correct(value.ownerId, request, now);
    expect(retry.duplicate).toBe(true);
    expect(retry.correctionCareEventId).toBe(first.correctionCareEventId);
    expect(retry.state.currentCareEventId).toBe(first.correctionCareEventId);

    const repaired = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM dogos_intelligence.observations
      WHERE source_event_id = ${first.correctionCareEventId}
        AND dimension = 'ENERGY'
    `);
    expect(repaired[0]?.count).toBe(1);
  });

  it('serializes different concurrent corrections so only one sequence-1 payload becomes canonical', async () => {
    const value = await fixture('race');
    const original = await capture(value, { appetite: 'USUAL' });

    const settled = await Promise.allSettled([
      correction.correct(
        value.ownerId,
        input(value, original.careEventId, { appetite: 'LESS' }),
        now
      ),
      correction.correct(
        value.memberId,
        input(value, original.careEventId, { appetite: 'MORE' }),
        now
      ),
    ]);

    expect(settled.filter((result) => result.status === 'fulfilled')).toHaveLength(1);
    expect(settled.filter((result) => result.status === 'rejected')).toHaveLength(1);

    const rows = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM care_events
      WHERE pet_id = ${value.petId}
        AND event_type = 'DAILY_SIGNALS_CORRECTION'
        AND context->>'rootCareEventId' = ${original.careEventId}
        AND context->>'correctionSequence' = '1'
    `);
    expect(rows[0]?.count).toBe(1);
  });

  it('allows an explicit clear-all correction and leaves zero effective baseline rows for that check-in', async () => {
    const value = await fixture('clear');
    const original = await capture(value, { energy: 'MORE', appetite: 'USUAL' });

    const receipt = await correction.correct(
      value.ownerId,
      input(value, original.careEventId, {}),
      now
    );

    expect(receipt.state.signals).toEqual({});
    expect(receipt.retractedDimensions).toEqual(expect.arrayContaining(['APPETITE', 'ENERGY']));

    const active = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM dogos_intelligence.observations o
      WHERE o.pet_id = ${value.petId}
        AND o.source_type = 'OWNER_CHECKIN'
        AND o.retracted_at IS NULL
        AND NOT EXISTS (
          SELECT 1
          FROM dogos_intelligence.observations successor
          WHERE successor.supersedes_observation_id = o.id
        )
    `);
    expect(active[0]?.count).toBe(0);
  });

  it('treats UNSURE as corrected uncertainty by retracting prior evidence without replacement', async () => {
    const value = await fixture('unsure');
    const original = await capture(value, { mobilityComfort: 'LESS' });

    const receipt = await correction.correct(
      value.ownerId,
      input(value, original.careEventId, { mobilityComfort: 'UNSURE' }),
      now
    );

    expect(receipt.state.signals).toEqual({ mobilityComfort: 'UNSURE' });
    expect(receipt.projectedDimensions).not.toContain('MOBILITY_COMFORT');
    expect(receipt.retractedDimensions).toContain('MOBILITY_COMFORT');
  });

  it('rejects a stale ancestor correction after the chain advances again', async () => {
    const value = await fixture('stale');
    const original = await capture(value, { energy: 'USUAL' });
    const first = await correction.correct(
      value.ownerId,
      input(value, original.careEventId, { energy: 'LESS' }),
      now
    );
    await correction.correct(
      value.ownerId,
      input(value, first.correctionCareEventId, { energy: 'MORE' }),
      now
    );

    await expect(
      correction.correct(
        value.ownerId,
        input(value, original.careEventId, { energy: 'LESS' }),
        now
      )
    ).rejects.toThrow('Reload before correcting');
  });

  it('denies current-state reads and corrections after household access is removed', async () => {
    const value = await fixture('authorization');
    const original = await capture(value, { energy: 'USUAL' });

    await prisma.householdMember.update({
      where: {
        householdId_userId: {
          householdId: value.householdId,
          userId: value.memberId,
        },
      },
      data: { status: 'INACTIVE' },
    });

    await expect(
      correction.getCurrent(
        value.memberId,
        { householdId: value.householdId, petId: value.petId },
        now
      )
    ).rejects.toThrow('Pet not found');
    await expect(
      correction.correct(
        value.memberId,
        input(value, original.careEventId, { energy: 'LESS' }),
        now
      )
    ).rejects.toThrow('Pet not found');
  });
});
