import { randomUUID } from 'node:crypto';
import { Prisma } from '@woof/database';
import { CareEventsService } from '../care-events/care-events.service';
import { HouseholdsService } from '../households/households.service';
import { PrismaService } from '../prisma/prisma.service';
import { DailySignalsService } from './daily-signals.service';
import type { CreateDailySignalsDto } from './dto/daily-signals.dto';
import { IntelligenceProjectionService } from './intelligence-projection.service';

type Fixture = {
  ownerId: string;
  memberId: string;
  petId: string;
  householdId: string;
};

describe('DailySignalsService integration', () => {
  const prisma = new PrismaService();
  const households = new HouseholdsService(prisma);
  const careEvents = new CareEventsService(prisma, households);
  const projection = new IntelligenceProjectionService(prisma, households);
  const service = new DailySignalsService(households, careEvents, projection);
  const usersToDelete: string[] = [];
  const householdsToDelete: string[] = [];

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
        handle: `daily-${label}-${suffix}`,
        email: `daily-${label}-${suffix}@example.test`,
      },
      select: { id: true },
    });
    usersToDelete.push(user.id);
    return user.id;
  }

  async function fixture(label: string, timezone: string | null = 'America/Los_Angeles') {
    const ownerId = await createUser(`${label}-owner`);
    const memberId = await createUser(`${label}-member`);
    const pet = await prisma.pet.create({
      data: {
        ownerId,
        name: `Daily ${label}`,
        species: 'DOG',
      },
      select: { id: true },
    });
    const householdId = await households.ensurePersonalHousehold(ownerId);
    householdsToDelete.push(householdId);

    await prisma.household.update({
      where: { id: householdId },
      data: { timezone },
    });
    await prisma.householdMember.upsert({
      where: { householdId_userId: { householdId, userId: memberId } },
      update: { status: 'ACTIVE', role: 'MEMBER' },
      create: { householdId, userId: memberId, status: 'ACTIVE', role: 'MEMBER' },
    });

    return { ownerId, memberId, petId: pet.id, householdId } satisfies Fixture;
  }

  function dto(
    fixtureValue: Pick<Fixture, 'petId' | 'householdId'>,
    overrides: Partial<CreateDailySignalsDto> = {}
  ): CreateDailySignalsDto {
    return {
      householdId: fixtureValue.householdId,
      petId: fixtureValue.petId,
      observedAt: '2026-08-25T18:00:00.000Z',
      signals: {
        appetite: 'USUAL',
        energy: 'MORE',
        bathroomRoutine: 'USUAL',
        mobilityComfort: 'LESS',
        engagementSocialComfort: 'USUAL',
        sleepRest: 'UNSURE',
      },
      note: 'A quiet morning after breakfast.',
      ...overrides,
    } as CreateDailySignalsDto;
  }

  it('converges 20 simultaneous cross-member retries on one private, zero-XP canonical event', async () => {
    const value = await fixture('cross-member-race');
    const input = dto(value);
    const now = new Date('2026-08-26T00:00:00.000Z');

    const receipts = await Promise.all(
      Array.from({ length: 20 }, (_, index) =>
        service.capture(index % 2 === 0 ? value.ownerId : value.memberId, input, now)
      )
    );

    expect(receipts.filter((receipt) => !receipt.duplicate)).toHaveLength(1);
    expect(receipts.filter((receipt) => receipt.duplicate)).toHaveLength(19);
    expect(new Set(receipts.map((receipt) => receipt.careEventId)).size).toBe(1);
    expect(new Set(receipts.map((receipt) => receipt.localDate)).size).toBe(1);

    const careEventId = receipts[0]!.careEventId;
    const eventRows = await prisma.$queryRaw<
      Array<{
        count: number;
        bond_xp: number;
        visibility: string;
        event_type: string;
        evidence_type: string | null;
      }>
    >(Prisma.sql`
      SELECT
        COUNT(*)::int AS count,
        COALESCE(MAX(rl.bond_xp), 0)::int AS bond_xp,
        MAX(ce.visibility) AS visibility,
        MAX(ce.event_type) AS event_type,
        MAX(ce.evidence_type) AS evidence_type
      FROM care_events ce
      LEFT JOIN reward_ledger rl ON rl.care_event_id = ce.id
      WHERE ce.id = ${careEventId}
    `);
    expect(eventRows[0]).toMatchObject({
      count: 1,
      bond_xp: 0,
      visibility: 'PRIVATE',
      event_type: 'DAILY_SIGNALS_CHECKIN',
      evidence_type: 'SELF_REPORT',
    });

    const sameDayRows = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM care_events
      WHERE pet_id = ${value.petId}
        AND event_type = 'DAILY_SIGNALS_CHECKIN'
        AND context->>'householdId' = ${value.householdId}
        AND context->>'localDate' = '2026-08-25'
    `);
    expect(sameDayRows[0]?.count).toBe(1);

    const projectionRows = await prisma.$queryRaw<
      Array<{ dimension: string; context: Record<string, unknown> }>
    >(Prisma.sql`
      SELECT dimension, context
      FROM dogos_intelligence.observations
      WHERE source_event_id = ${careEventId}
      ORDER BY dimension
    `);
    expect(projectionRows).toHaveLength(5);
    expect(projectionRows.map((row) => row.dimension)).not.toContain('SLEEP_REST');
    expect(projectionRows.every((row) => !('note' in row.context))).toBe(true);
  });

  it('makes the first accepted same-day payload canonical and conflicts on different answers', async () => {
    const value = await fixture('payload-conflict');
    const now = new Date('2026-08-26T00:00:00.000Z');
    const ownerInput = dto(value, {
      signals: { appetite: 'USUAL', energy: 'USUAL' },
      note: 'Owner view',
    });
    const memberInput = dto(value, {
      signals: { appetite: 'MORE', energy: 'USUAL' },
      note: 'Member view',
    });

    const settled = await Promise.allSettled([
      service.capture(value.ownerId, ownerInput, now),
      service.capture(value.memberId, memberInput, now),
    ]);
    expect(settled.filter((result) => result.status === 'fulfilled')).toHaveLength(1);
    expect(settled.filter((result) => result.status === 'rejected')).toHaveLength(1);

    const rejected = settled.find((result) => result.status === 'rejected');
    expect(rejected && rejected.status === 'rejected' ? String(rejected.reason) : '').toContain(
      'different answers'
    );

    const rows = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM care_events
      WHERE pet_id = ${value.petId}
        AND event_type = 'DAILY_SIGNALS_CHECKIN'
    `);
    expect(rows[0]?.count).toBe(1);
  });

  it('keeps two pets isolated on the same household-local day', async () => {
    const value = await fixture('two-pet-isolation');
    const secondPet = await prisma.pet.create({
      data: {
        ownerId: value.ownerId,
        name: 'Daily second pet',
        species: 'DOG',
      },
      select: { id: true },
    });
    await households.addOwnedPet(value.ownerId, value.householdId, secondPet.id);
    const now = new Date('2026-08-26T00:00:00.000Z');

    const first = await service.capture(value.ownerId, dto(value), now);
    const second = await service.capture(
      value.ownerId,
      dto({ householdId: value.householdId, petId: secondPet.id }),
      now
    );

    expect(first.careEventId).not.toBe(second.careEventId);
    expect(first.localDate).toBe(second.localDate);

    const rows = await prisma.$queryRaw<Array<{ pet_id: string }>>(Prisma.sql`
      SELECT pet_id
      FROM care_events
      WHERE event_type = 'DAILY_SIGNALS_CHECKIN'
        AND context->>'householdId' = ${value.householdId}
        AND context->>'localDate' = '2026-08-25'
      ORDER BY pet_id
    `);
    expect(new Set(rows.map((row) => row.pet_id))).toEqual(new Set([value.petId, secondPet.id]));
  });

  it('repairs a partially missing projection from the canonical CareEvent on retry', async () => {
    const value = await fixture('replay-repair');
    const input = dto(value, {
      signals: { appetite: 'USUAL', energy: 'MORE', mobilityComfort: 'LESS' },
      note: undefined,
    });
    const now = new Date('2026-08-26T00:00:00.000Z');
    const first = await service.capture(value.ownerId, input, now);

    await prisma.$executeRaw(Prisma.sql`
      DELETE FROM dogos_intelligence.observations
      WHERE source_event_id = ${first.careEventId}
        AND dimension = 'ENERGY'
    `);
    const afterDelete = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM dogos_intelligence.observations
      WHERE source_event_id = ${first.careEventId}
    `);
    expect(afterDelete[0]?.count).toBe(2);

    const retry = await service.capture(value.ownerId, input, now);
    expect(retry.duplicate).toBe(true);
    expect(retry.careEventId).toBe(first.careEventId);

    const repaired = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM dogos_intelligence.observations
      WHERE source_event_id = ${first.careEventId}
    `);
    expect(repaired[0]?.count).toBe(3);
  });

  it('clamps future timestamps before choosing the household-local identity', async () => {
    const value = await fixture('future-time');
    const now = new Date();
    const requested = new Date(now.getTime() + 7 * 24 * 60 * 60 * 1000).toISOString();
    const input = dto(value, { observedAt: requested, signals: { appetite: 'USUAL' } });

    const receipt = await service.capture(value.ownerId, input, now);
    expect(receipt.futureTimestampNormalized).toBe(true);
    expect(receipt.timezone).toBe('America/Los_Angeles');

    const canonical = await careEvents.getAuthorizedEvent(value.ownerId, receipt.careEventId);
    expect(canonical.context.localDate).toBe(receipt.localDate);
    expect(new Date(canonical.occurredAt).getTime()).toBeLessThanOrEqual(Date.now() + 1000);
  });

  it('fails closed for missing or invalid household timezone configuration', async () => {
    const missing = await fixture('missing-zone', null);
    await expect(service.capture(missing.ownerId, dto(missing))).rejects.toThrow(
      'Household timezone is required'
    );

    const invalid = await fixture('invalid-zone');
    await prisma.household.update({
      where: { id: invalid.householdId },
      data: { timezone: 'Mars/Olympus' },
    });
    await expect(service.capture(invalid.ownerId, dto(invalid))).rejects.toThrow(
      'valid IANA timezone'
    );
  });

  it('rejects unrelated and removed household members without a pet-existence leak', async () => {
    const value = await fixture('authorization');
    const outsiderId = await createUser('outsider');

    await expect(service.capture(outsiderId, dto(value))).rejects.toThrow('Pet not found');

    await prisma.householdMember.update({
      where: {
        householdId_userId: {
          householdId: value.householdId,
          userId: value.memberId,
        },
      },
      data: { status: 'INACTIVE' },
    });
    await expect(service.capture(value.memberId, dto(value))).rejects.toThrow('Pet not found');

    const ownerReceipt = await service.capture(value.ownerId, dto(value));
    expect(ownerReceipt.petId).toBe(value.petId);
  });

  it('rejects invalid direct-service payloads', async () => {
    const value = await fixture('payload-validation');

    await expect(
      service.capture(
        value.ownerId,
        dto(value, { signals: {} }),
        new Date('2026-08-26T00:00:00.000Z')
      )
    ).rejects.toThrow('at least one answered dimension');
    await expect(
      service.capture(
        value.ownerId,
        dto(value, { note: 'x'.repeat(501) }),
        new Date('2026-08-26T00:00:00.000Z')
      )
    ).rejects.toThrow('500 characters or fewer');
  });

  it('validates newly configured household timezones before persistence', async () => {
    const value = await fixture('timezone-update');
    await expect(
      households.update(value.ownerId, value.householdId, { timezone: 'Mars/Olympus' })
    ).rejects.toThrow('valid IANA timezone');

    const updated = await households.update(value.ownerId, value.householdId, {
      timezone: 'Asia/Tokyo',
    });
    expect(updated.timezone).toBe('Asia/Tokyo');
  });

  it('fails closed when asked to replay an orphan source ID', async () => {
    const value = await fixture('orphan-replay');
    await expect(service.replay(value.ownerId, randomUUID())).rejects.toThrow(
      'CareEvent not found'
    );
  });
});
