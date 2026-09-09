import { randomUUID } from 'node:crypto';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { PackAccessService } from '../social-adventure/pack-access.service';
import { currentExpeditionSeason } from './expeditions.policy';
import { ExpeditionsService } from './expeditions.service';

type ExpeditionResponse = Awaited<ReturnType<ExpeditionsService['getGlobal']>>;

describe('ExpeditionsService integration', () => {
  const prisma = new PrismaService();
  const packAccess = new PackAccessService(prisma);
  const service = new ExpeditionsService(prisma, packAccess);
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
        handle: `expedition-${label}-${suffix}`,
        email: `expedition-${label}-${suffix}@example.test`,
      },
      select: { id: true },
    });
    usersToDelete.push(user.id);
    return user.id;
  }

  async function insertCareEvent(
    userId: string,
    pathway: 'EXPLORE' | 'ENRICH' | 'RECOVER' | 'CARE',
    occurredAt: Date,
    index: number
  ) {
    const id = randomUUID();
    await prisma.$executeRaw(Prisma.sql`
      INSERT INTO public.care_events (
        id,
        user_id,
        pet_id,
        event_type,
        pathway,
        occurred_at,
        source,
        evidence_confidence,
        dedupe_key,
        visibility
      ) VALUES (
        ${id},
        ${userId},
        NULL,
        ${`QUEST_${pathway}`},
        ${pathway},
        ${occurredAt},
        'QUEST_ENGINE',
        0.8,
        ${`expedition-test:${userId}:${pathway}:${index}:${id}`},
        'PRIVATE'
      )
    `);
    return id;
  }

  async function insertHumanSkill(
    userId: string,
    challengeKey: 'MAKE_IT_EASIER' | 'MARKER_TIMING',
    completedAt: Date,
    score: number
  ) {
    const id = randomUUID();
    const issuedAt = new Date(completedAt.getTime() - 60_000);
    const expiresAt = new Date(completedAt.getTime() + 9 * 60_000);
    await prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.human_skill_attempts (
        id,
        user_id,
        challenge_key,
        challenge_version,
        scenario_key,
        issued_at,
        expires_at,
        completed_at,
        response,
        score,
        receipt
      ) VALUES (
        ${id},
        ${userId},
        ${challengeKey},
        'human-skill-arcade-v1',
        ${`expedition-${challengeKey.toLowerCase()}`},
        ${issuedAt},
        ${expiresAt},
        ${completedAt},
        '{"test":true}'::jsonb,
        ${score},
        ${JSON.stringify({ score, test: true })}::jsonb
      )
    `);
    return id;
  }

  function objective(response: ExpeditionResponse, key: string) {
    const found = response.objectives.find((item) => item.key === key);
    if (!found) throw new Error(`Missing Expedition objective ${key}`);
    return found;
  }

  it('materializes bounded Global receipts from canonical evidence without CARE or score magnitude', async () => {
    const userId = await createUser('global');
    const season = currentExpeditionSeason();
    const evidenceAt = new Date(season.startsAt.getTime() + 6 * 60 * 60 * 1000);

    for (let index = 0; index < 3; index += 1) {
      await insertCareEvent(userId, 'EXPLORE', new Date(evidenceAt.getTime() + index * 1000), index);
      await insertCareEvent(
        userId,
        'ENRICH',
        new Date(evidenceAt.getTime() + 10_000 + index * 1000),
        index
      );
      await insertCareEvent(
        userId,
        'RECOVER',
        new Date(evidenceAt.getTime() + 20_000 + index * 1000),
        index
      );
    }
    await insertCareEvent(userId, 'CARE', new Date(evidenceAt.getTime() + 30_000), 0);

    await insertHumanSkill(userId, 'MAKE_IT_EASIER', new Date(evidenceAt.getTime() + 40_000), 1);
    await insertHumanSkill(userId, 'MAKE_IT_EASIER', new Date(evidenceAt.getTime() + 41_000), 100);
    await insertHumanSkill(userId, 'MARKER_TIMING', new Date(evidenceAt.getTime() + 42_000), 5);

    const projection = await service.getGlobal(userId);
    expect(objective(projection, 'SNIFF_EXPLORE').myContribution).toBe(4);
    expect(objective(projection, 'RECOVERY_COUNTS').myContribution).toBe(2);
    expect(objective(projection, 'READ_THE_ROOM').myContribution).toBe(2);
    expect(projection.objectives.every((item) => item.status === 'CALIBRATING')).toBe(true);
    expect(projection.objectives.every((item) => item.target === null)).toBe(true);

    await Promise.all(Array.from({ length: 5 }, () => service.getGlobal(userId)));
    const receiptRows = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM dogos_social.expedition_receipts
      WHERE user_id = ${userId} AND scope = 'GLOBAL'
    `);
    expect(receiptRows[0]?.count).toBe(8);

    const careRows = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM dogos_social.expedition_receipts
      WHERE user_id = ${userId} AND pathway = 'CARE'
    `);
    expect(careRows[0]?.count).toBe(0);
  });

  it('requires ACTIVE Pack membership and keeps season caps across leave and rejoin', async () => {
    const ownerId = await createUser('pack-owner');
    const memberId = await createUser('pack-member');
    const formerMemberId = await createUser('pack-former');
    const outsiderId = await createUser('pack-outsider');
    const packId = randomUUID();
    const season = currentExpeditionSeason();
    const joinedAt = new Date(season.startsAt.getTime() + 2 * 60 * 60 * 1000);
    const beforeJoin = new Date(joinedAt.getTime() - 30 * 60 * 1000);
    const afterJoin = new Date(joinedAt.getTime() + 30 * 60 * 1000);

    await prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.packs (
        id, owner_user_id, name, slug, scope, region_key, visibility, created_at
      ) VALUES (
        ${packId}, ${ownerId}, 'Expedition Test Pack', ${`expedition-test-${packId}`},
        'LOCAL', 'test-region', 'PUBLIC', ${season.startsAt}
      )
    `);
    await prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.pack_memberships (pack_id, user_id, role, status, joined_at)
      VALUES
        (${packId}, ${ownerId}, 'OWNER', 'ACTIVE', ${joinedAt}),
        (${packId}, ${memberId}, 'MEMBER', 'ACTIVE', ${joinedAt}),
        (${packId}, ${formerMemberId}, 'MEMBER', 'LEFT', ${joinedAt})
    `);

    await insertCareEvent(ownerId, 'EXPLORE', beforeJoin, 0);
    await insertCareEvent(ownerId, 'EXPLORE', afterJoin, 1);
    await insertCareEvent(memberId, 'ENRICH', afterJoin, 0);
    await insertCareEvent(formerMemberId, 'EXPLORE', afterJoin, 0);
    await insertCareEvent(outsiderId, 'EXPLORE', afterJoin, 0);
    await insertHumanSkill(memberId, 'MAKE_IT_EASIER', new Date(afterJoin.getTime() + 60_000), 1);

    await expect(service.getPack(outsiderId, packId)).rejects.toThrow('Pack not found');

    const packProjection = await service.getPack(ownerId, packId);
    expect(objective(packProjection, 'SNIFF_EXPLORE').total).toBe(2);
    expect(objective(packProjection, 'SNIFF_EXPLORE').myContribution).toBe(1);
    expect(objective(packProjection, 'READ_THE_ROOM').total).toBe(1);

    await Promise.all(Array.from({ length: 5 }, () => service.getPack(ownerId, packId)));
    const beforeLeaveRows = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM dogos_social.expedition_receipts
      WHERE scope = 'PACK' AND pack_id = ${packId}
    `);
    expect(beforeLeaveRows[0]?.count).toBe(3);

    await prisma.$executeRaw(Prisma.sql`
      UPDATE dogos_social.pack_memberships
      SET status = 'LEFT'
      WHERE pack_id = ${packId} AND user_id = ${memberId}
    `);
    const afterLeave = await service.getPack(ownerId, packId);
    expect(objective(afterLeave, 'SNIFF_EXPLORE').total).toBe(2);
    expect(objective(afterLeave, 'READ_THE_ROOM').total).toBe(1);

    const rejoinedAt = new Date(afterJoin.getTime() + 2 * 60 * 60 * 1000);
    await prisma.$executeRaw(Prisma.sql`
      UPDATE dogos_social.pack_memberships
      SET status = 'ACTIVE', joined_at = ${rejoinedAt}
      WHERE pack_id = ${packId} AND user_id = ${memberId}
    `);
    await insertCareEvent(memberId, 'ENRICH', new Date(rejoinedAt.getTime() + 60_000), 1);
    await insertCareEvent(memberId, 'ENRICH', new Date(rejoinedAt.getTime() + 120_000), 2);
    await insertHumanSkill(memberId, 'MAKE_IT_EASIER', new Date(rejoinedAt.getTime() + 180_000), 100);

    const afterRejoin = await service.getPack(ownerId, packId);
    expect(objective(afterRejoin, 'SNIFF_EXPLORE').total).toBe(3);
    expect(objective(afterRejoin, 'READ_THE_ROOM').total).toBe(1);

    const memberCategoryRows = await prisma.$queryRaw<
      Array<{ sourceType: string; categoryKey: string; count: number }>
    >(Prisma.sql`
      SELECT
        source_type AS "sourceType",
        category_key AS "categoryKey",
        COUNT(*)::int AS count
      FROM dogos_social.expedition_receipts
      WHERE scope = 'PACK'
        AND pack_id = ${packId}
        AND user_id = ${memberId}
      GROUP BY source_type, category_key
      ORDER BY source_type, category_key
    `);
    expect(memberCategoryRows).toEqual([
      { sourceType: 'CARE_EVENT', categoryKey: 'ENRICH', count: 2 },
      { sourceType: 'HUMAN_SKILL_ATTEMPT', categoryKey: 'MAKE_IT_EASIER', count: 1 },
    ]);

    const receipt = await prisma.$queryRaw<Array<{ id: string }>>(Prisma.sql`
      SELECT id
      FROM dogos_social.expedition_receipts
      WHERE scope = 'PACK' AND pack_id = ${packId}
      LIMIT 1
    `);
    expect(receipt[0]?.id).toBeDefined();
    await expect(
      prisma.$executeRaw(Prisma.sql`
        UPDATE dogos_social.expedition_receipts
        SET authorized_at = authorized_at + INTERVAL '1 second'
        WHERE id = ${receipt[0]?.id}
      `)
    ).rejects.toThrow();
  });
});
