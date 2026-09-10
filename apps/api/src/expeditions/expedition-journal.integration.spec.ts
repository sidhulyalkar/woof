import { createHash, randomUUID } from 'node:crypto';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { PackAccessService } from '../social-adventure/pack-access.service';
import { ExpeditionJournalService } from './expedition-journal.service';
import { currentExpeditionSeason } from './expeditions.policy';
import { ExpeditionsService } from './expeditions.service';

describe('ExpeditionJournalService integration', () => {
  const prisma = new PrismaService();
  const packAccess = new PackAccessService(prisma);
  const expeditions = new ExpeditionsService(prisma, packAccess);
  const journal = new ExpeditionJournalService(prisma, expeditions);
  const usersToDelete: string[] = [];
  const packsToDelete: string[] = [];

  beforeAll(async () => {
    await prisma.$connect();
  });

  afterAll(async () => {
    for (const packId of packsToDelete) {
      await prisma.$executeRaw(Prisma.sql`DELETE FROM dogos_social.packs WHERE id = ${packId}`);
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
        handle: `journal-${label}-${suffix}`,
        email: `journal-${label}-${suffix}@example.test`,
      },
      select: { id: true },
    });
    usersToDelete.push(user.id);
    return user.id;
  }

  async function insertCareEvent(userId: string, occurredAt: Date, index: number) {
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
        'QUEST_EXPLORE',
        'EXPLORE',
        ${occurredAt},
        'QUEST_ENGINE',
        0.8,
        ${`journal-test:${userId}:EXPLORE:${index}:${id}`},
        'PRIVATE'
      )
    `);
  }

  async function insertHumanSkill(userId: string, completedAt: Date) {
    const id = randomUUID();
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
        'MAKE_IT_EASIER',
        'human-skill-arcade-v1',
        'journal-make-it-easier',
        ${new Date(completedAt.getTime() - 60_000)},
        ${new Date(completedAt.getTime() + 9 * 60_000)},
        ${completedAt},
        '{"test":true}'::jsonb,
        100,
        '{"score":100,"test":true}'::jsonb
      )
    `);
  }

  async function insertReceipt(input: {
    userId: string;
    seasonKey: string;
    scope: 'GLOBAL' | 'PACK';
    packId?: string | null;
    objectiveKey: 'SNIFF_EXPLORE' | 'RECOVERY_COUNTS' | 'READ_THE_ROOM';
    categoryKey: string;
    sourceType?: 'CARE_EVENT' | 'HUMAN_SKILL_ATTEMPT';
    pathway?: 'EXPLORE' | 'ENRICH' | 'RECOVER' | null;
    suffix: string;
  }) {
    const sourceType = input.sourceType ?? 'CARE_EVENT';
    const sourceId = `journal-source-${input.suffix}-${randomUUID()}`;
    const identity = `${input.userId}|${input.seasonKey}|${input.scope}|${input.objectiveKey}|${sourceId}`;
    const fingerprint = createHash('sha256').update(identity).digest('hex');
    await prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.expedition_receipts (
        id,
        expedition_key,
        expedition_version,
        season_key,
        policy_version,
        scope,
        pack_id,
        user_id,
        source_type,
        source_id,
        objective_key,
        category_key,
        pathway,
        source_fingerprint,
        evidence_at
      ) VALUES (
        ${`exp:${fingerprint}`},
        'shared-world',
        'v1',
        ${input.seasonKey},
        'expedition-authority-v1',
        ${input.scope},
        ${input.packId ?? null},
        ${input.userId},
        ${sourceType},
        ${sourceId},
        ${input.objectiveKey},
        ${input.categoryKey},
        ${input.pathway ?? null},
        ${fingerprint},
        NOW()
      )
    `);
  }

  it('projects personal Global landmark presence without volume, Pack, future, or malformed-season leakage', async () => {
    const userId = await createUser('owner');
    const otherUserId = await createUser('other');
    const season = currentExpeditionSeason();
    const evidenceAt = new Date(season.startsAt.getTime() + 6 * 60 * 60 * 1000);

    await insertCareEvent(userId, evidenceAt, 0);
    await insertCareEvent(userId, new Date(evidenceAt.getTime() + 1000), 1);
    await insertHumanSkill(userId, new Date(evidenceAt.getTime() + 2000));

    const previousStart = new Date(season.startsAt.getTime() - 7 * 24 * 60 * 60 * 1000);
    const previousKey = `week:${previousStart.toISOString().slice(0, 10)}`;
    await insertReceipt({
      userId,
      seasonKey: previousKey,
      scope: 'GLOBAL',
      objectiveKey: 'RECOVERY_COUNTS',
      categoryKey: 'RECOVER',
      pathway: 'RECOVER',
      suffix: 'previous-recovery',
    });
    await insertReceipt({
      userId: otherUserId,
      seasonKey: previousKey,
      scope: 'GLOBAL',
      objectiveKey: 'READ_THE_ROOM',
      categoryKey: 'MARKER_TIMING',
      sourceType: 'HUMAN_SKILL_ATTEMPT',
      pathway: null,
      suffix: 'other-user-skill',
    });

    const packId = randomUUID();
    packsToDelete.push(packId);
    await prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.packs (
        id, owner_user_id, name, slug, scope, region_key, visibility, created_at
      ) VALUES (
        ${packId},
        ${userId},
        'Journal Test Pack',
        ${`journal-test-${packId}`},
        'LOCAL',
        'test-region',
        'PUBLIC',
        ${season.startsAt}
      )
    `);
    await insertReceipt({
      userId,
      seasonKey: previousKey,
      scope: 'PACK',
      packId,
      objectiveKey: 'SNIFF_EXPLORE',
      categoryKey: 'EXPLORE',
      pathway: 'EXPLORE',
      suffix: 'pack-only',
    });

    const futureStart = new Date(season.startsAt.getTime() + 7 * 24 * 60 * 60 * 1000);
    await insertReceipt({
      userId,
      seasonKey: `week:${futureStart.toISOString().slice(0, 10)}`,
      scope: 'GLOBAL',
      objectiveKey: 'SNIFF_EXPLORE',
      categoryKey: 'EXPLORE',
      pathway: 'EXPLORE',
      suffix: 'future',
    });

    const malformedStart = new Date(season.startsAt.getTime() - 5 * 24 * 60 * 60 * 1000);
    await insertReceipt({
      userId,
      seasonKey: `week:${malformedStart.toISOString().slice(0, 10)}`,
      scope: 'GLOBAL',
      objectiveKey: 'READ_THE_ROOM',
      categoryKey: 'PAIRING_LAB',
      sourceType: 'HUMAN_SKILL_ATTEMPT',
      pathway: null,
      suffix: 'non-monday',
    });

    const result = await journal.getMine(userId);

    expect(result.scope).toBe('GLOBAL');
    expect(result.coverage).toEqual({ kind: 'RECENT_PARTICIPATED_SEASONS', maxSeasons: 26 });
    expect(result.entries.map((entry) => entry.season.key)).toEqual([season.key, previousKey]);

    const current = result.entries[0];
    expect(current?.state).toBe('ACTIVE');
    expect(current?.landmarks).toEqual([
      { key: 'SNIFF_EXPLORE', title: 'Sniff & Explore' },
      { key: 'READ_THE_ROOM', title: 'Read the Room' },
    ]);

    const previous = result.entries[1];
    expect(previous?.state).toBe('PAST');
    expect(previous?.landmarks).toEqual([
      { key: 'RECOVERY_COUNTS', title: 'Recovery Counts' },
    ]);

    expect(Object.keys(current ?? {}).sort()).toEqual(['landmarks', 'season', 'state']);
    expect(Object.keys(current?.landmarks[0] ?? {}).sort()).toEqual(['key', 'title']);
    expect(result.entries.some((entry) => entry.season.key.includes(futureStart.toISOString().slice(0, 10)))).toBe(false);
    expect(result.entries.some((entry) => entry.season.key.includes(malformedStart.toISOString().slice(0, 10)))).toBe(false);
  });
});
