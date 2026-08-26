import { randomUUID } from 'node:crypto';
import { Prisma } from '@woof/database';
import { HouseholdsService } from '../households/households.service';
import { PrismaService } from '../prisma/prisma.service';
import { CareEventsService } from './care-events.service';
import { rewardCareEvent } from './reward-policy';

const emptyPolicyContext = {
  totalXpToday: 0,
  pathwayXpToday: 0,
  samePathwayEventsToday: 0,
  repeatedEventCount7d: 0,
};

type Fixture = {
  userId: string;
  petId: string;
};

describe('CareEventsService integration', () => {
  const prisma = new PrismaService();
  const households = new HouseholdsService(prisma);
  const service = new CareEventsService(prisma, households);
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
        handle: `adventure-${label}-${suffix}`,
        email: `adventure-${label}-${suffix}@example.test`,
      },
      select: { id: true },
    });
    usersToDelete.push(user.id);
    return user;
  }

  async function fixture(label: string): Promise<Fixture> {
    const user = await createUser(label);
    const pet = await prisma.pet.create({
      data: {
        ownerId: user.id,
        name: `Test ${label}`,
        species: 'DOG',
      },
      select: { id: true },
    });

    return { userId: user.id, petId: pet.id };
  }

  it('issues exactly one reward for simultaneous requests with the same dedupe key', async () => {
    const { userId, petId } = await fixture('dedupe');
    const input = {
      userId,
      petId,
      eventType: 'QUEST_EXPLORE',
      pathway: 'EXPLORE' as const,
      source: 'INTEGRATION_TEST',
      evidenceType: 'SELF_REPORT' as const,
      evidenceConfidence: 0.65,
      dedupeKey: `race:${randomUUID()}`,
    };

    const receipts = await Promise.all(Array.from({ length: 10 }, () => service.record(input)));
    const newlyIssued = receipts.filter((receipt) => !receipt.duplicate);
    const duplicates = receipts.filter((receipt) => receipt.duplicate);

    expect(newlyIssued).toHaveLength(1);
    expect(duplicates).toHaveLength(9);
    expect(new Set(receipts.map((receipt) => receipt.careEventId)).size).toBe(1);
    expect(new Set(receipts.map((receipt) => receipt.ledgerId)).size).toBe(1);

    const eventRows = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM care_events
      WHERE user_id = ${userId} AND dedupe_key = ${input.dedupeKey}
    `);
    const ledgerRows = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM reward_ledger
      WHERE user_id = ${userId}
    `);
    const user = await prisma.user.findUniqueOrThrow({
      where: { id: userId },
      select: { totalPoints: true },
    });

    expect(eventRows[0]?.count).toBe(1);
    expect(ledgerRows[0]?.count).toBe(1);
    expect(user.totalPoints).toBe(newlyIssued[0]?.bondXp);
  });

  it('serializes distinct simultaneous rewards so a pathway cap cannot be raced', async () => {
    const { userId, petId } = await fixture('cap-race');

    const receipts = await Promise.all(
      Array.from({ length: 12 }, (_, index) =>
        service.record({
          userId,
          petId,
          eventType: 'ACTIVITY_HIKE',
          pathway: 'MOVE',
          source: 'INTEGRATION_TEST',
          evidenceType: 'ACTIVITY',
          evidenceConfidence: 0.9,
          dedupeKey: `cap-race:${index}:${randomUUID()}`,
        })
      )
    );

    const issuedXp = receipts.reduce((sum, receipt) => sum + receipt.bondXp, 0);
    const ledgerRows = await prisma.$queryRaw<Array<{ xp: number }>>(Prisma.sql`
      SELECT COALESCE(SUM(bond_xp), 0)::int AS xp
      FROM reward_ledger
      WHERE user_id = ${userId}
    `);

    expect(receipts.every((receipt) => !receipt.duplicate)).toBe(true);
    expect(issuedXp).toBeLessThanOrEqual(60);
    expect(ledgerRows[0]?.xp).toBe(issuedXp);
  });

  it('does not let a future occurrence timestamp bypass caps or skew recent summaries', async () => {
    const { userId, petId } = await fixture('future-time');

    for (let index = 0; index < 12; index += 1) {
      await service.record({
        userId,
        petId,
        eventType: 'ACTIVITY_HIKE',
        pathway: 'MOVE',
        source: 'INTEGRATION_TEST',
        evidenceType: 'ACTIVITY',
        evidenceConfidence: 0.9,
        dedupeKey: `future-fill:${index}:${randomUUID()}`,
      });
    }

    const before = await prisma.$queryRaw<Array<{ xp: number }>>(Prisma.sql`
      SELECT COALESCE(SUM(bond_xp), 0)::int AS xp
      FROM reward_ledger
      WHERE user_id = ${userId}
    `);
    expect(before[0]?.xp).toBe(60);

    const beforeAttemptMs = Date.now();
    const futureReceipt = await service.record({
      userId,
      petId,
      eventType: 'ACTIVITY_HIKE',
      pathway: 'MOVE',
      occurredAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
      source: 'INTEGRATION_TEST',
      evidenceType: 'ACTIVITY',
      evidenceConfidence: 0.9,
      dedupeKey: `future-attempt:${randomUUID()}`,
    });

    expect(futureReceipt.bondXp).toBe(0);

    const futureEventRows = await prisma.$queryRaw<Array<{ occurred_at: Date }>>(Prisma.sql`
      SELECT occurred_at
      FROM care_events
      WHERE id = ${futureReceipt.careEventId}
      LIMIT 1
    `);
    const storedOccurrenceMs = futureEventRows[0]?.occurred_at.getTime();
    expect(storedOccurrenceMs).toBeDefined();
    expect(storedOccurrenceMs).toBeGreaterThanOrEqual(beforeAttemptMs - 1000);
    expect(storedOccurrenceMs).toBeLessThanOrEqual(Date.now() + 1000);

    const summary = await service.getSummary(userId, petId);
    const moveSummary = summary.pathways.find((pathway) => pathway.pathway === 'MOVE');
    expect(moveSummary?.lastEventAt).not.toBeNull();
    expect(new Date(moveSummary!.lastEventAt!).getTime()).toBeLessThanOrEqual(Date.now() + 1000);
  });

  it('does not let a zero-XP safety event decay the next legitimate reward', async () => {
    const { userId, petId } = await fixture('safety-zero');
    const common = {
      userId,
      petId,
      eventType: 'QUEST_EXPLORE',
      pathway: 'EXPLORE' as const,
      source: 'INTEGRATION_TEST',
      evidenceType: 'SELF_REPORT' as const,
      evidenceConfidence: 0.65,
    };

    const blocked = await service.record({
      ...common,
      dedupeKey: `blocked:${randomUUID()}`,
      safetyEligible: false,
    });
    expect(blocked.bondXp).toBe(0);

    const legitimate = await service.record({
      ...common,
      dedupeKey: `legitimate:${randomUUID()}`,
      safetyEligible: true,
    });
    const baseline = rewardCareEvent(
      {
        ...common,
        dedupeKey: 'pure-baseline',
        safetyEligible: true,
      },
      emptyPolicyContext
    );

    expect(legitimate.bondXp).toBe(baseline.bondXp);
  });

  it('uses household pet authority for canonical zero-reward Daily Signals evidence', async () => {
    const { userId: ownerId, petId } = await fixture('household-owner');
    const member = await createUser('household-member');
    const outsider = await createUser('household-outsider');
    const householdId = await households.ensurePersonalHousehold(ownerId);

    await prisma.householdMember.upsert({
      where: {
        householdId_userId: {
          householdId,
          userId: member.id,
        },
      },
      update: { status: 'ACTIVE', role: 'MEMBER' },
      create: {
        householdId,
        userId: member.id,
        status: 'ACTIVE',
        role: 'MEMBER',
      },
    });

    const dedupeKey = `daily-signals:${petId}:2026-08-25:${randomUUID()}`;
    const receipt = await service.record({
      userId: member.id,
      petId,
      eventType: 'DAILY_SIGNALS_CHECKIN',
      pathway: 'CARE',
      occurredAt: new Date('2026-08-25T18:00:00.000Z'),
      source: 'INTELLIGENCE',
      evidenceType: 'SELF_REPORT',
      evidenceConfidence: 0.8,
      dedupeKey,
      visibility: 'PRIVATE',
      safetyEligible: false,
      context: { localDate: '2026-08-25', timezone: 'America/Los_Angeles' },
      outcome: { APPETITE: 'USUAL' },
    });

    expect(receipt.bondXp).toBe(0);
    expect(receipt.duplicate).toBe(false);

    const stored = await prisma.$queryRaw<
      Array<{ user_id: string; pet_id: string; visibility: string; evidence_type: string }>
    >(Prisma.sql`
      SELECT user_id, pet_id, visibility, evidence_type
      FROM care_events
      WHERE id = ${receipt.careEventId}
    `);
    expect(stored[0]).toEqual({
      user_id: member.id,
      pet_id: petId,
      visibility: 'PRIVATE',
      evidence_type: 'SELF_REPORT',
    });

    await expect(
      service.record({
        userId: outsider.id,
        petId,
        eventType: 'DAILY_SIGNALS_CHECKIN',
        pathway: 'CARE',
        source: 'INTELLIGENCE',
        evidenceType: 'SELF_REPORT',
        dedupeKey: `outsider:${randomUUID()}`,
        safetyEligible: false,
      })
    ).rejects.toThrow('Pet not found');
  });
});
