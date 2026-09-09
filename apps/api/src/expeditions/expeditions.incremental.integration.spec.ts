import { randomUUID } from 'node:crypto';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { PackAccessService } from '../social-adventure/pack-access.service';
import { currentExpeditionSeason } from './expeditions.policy';
import { ExpeditionsService } from './expeditions.service';

describe('ExpeditionsService incremental Global reconciliation', () => {
  const prisma = new PrismaService();
  const service = new ExpeditionsService(prisma, new PackAccessService(prisma));
  let userId: string | null = null;

  beforeAll(async () => {
    await prisma.$connect();
  });

  afterAll(async () => {
    if (userId) {
      await prisma.user.deleteMany({ where: { id: userId } });
    }
    await prisma.$disconnect();
  });

  async function insertExplore(occurredAt: Date, index: number) {
    if (!userId) throw new Error('Test user not initialized');
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
        ${`expedition-incremental:${index}:${id}`},
        'PRIVATE'
      )
    `);
  }

  it('uses already-issued receipts to preserve a category cap as new evidence arrives', async () => {
    const suffix = randomUUID().slice(0, 8);
    const user = await prisma.user.create({
      data: {
        handle: `expedition-incremental-${suffix}`,
        email: `expedition-incremental-${suffix}@example.test`,
      },
      select: { id: true },
    });
    userId = user.id;

    const season = currentExpeditionSeason();
    const start = new Date(season.startsAt.getTime() + 4 * 60 * 60 * 1000);
    await insertExplore(start, 0);

    const first = await service.getGlobal(user.id);
    const firstObjective = first.objectives.find((item) => item.key === 'SNIFF_EXPLORE');
    expect(firstObjective?.myContribution).toBe(1);

    await insertExplore(new Date(start.getTime() + 60_000), 1);
    await insertExplore(new Date(start.getTime() + 120_000), 2);

    await Promise.all(Array.from({ length: 5 }, () => service.getGlobal(user.id)));
    const final = await service.getGlobal(user.id);
    const finalObjective = final.objectives.find((item) => item.key === 'SNIFF_EXPLORE');
    expect(finalObjective?.myContribution).toBe(2);

    const rows = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM dogos_social.expedition_receipts
      WHERE user_id = ${user.id}
        AND scope = 'GLOBAL'
        AND source_type = 'CARE_EVENT'
        AND category_key = 'EXPLORE'
    `);
    expect(rows[0]?.count).toBe(2);
  });
});
