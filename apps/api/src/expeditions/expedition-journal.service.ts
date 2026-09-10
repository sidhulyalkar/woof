import { Injectable } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import {
  currentExpeditionSeason,
  EXPEDITION_KEY,
  EXPEDITION_OBJECTIVES,
  EXPEDITION_POLICY_VERSION,
  EXPEDITION_VERSION,
  type ExpeditionObjectiveKey,
} from './expeditions.policy';
import { ExpeditionsService } from './expeditions.service';

const JOURNAL_MAX_PARTICIPATED_SEASONS = 26;

type JournalRow = {
  seasonKey: string;
  objectiveKey: string;
};

type JournalSeason = {
  key: string;
  startsAt: Date;
  endsAt: Date;
};

@Injectable()
export class ExpeditionJournalService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly expeditions: ExpeditionsService
  ) {}

  async getMine(userId: string) {
    // Keep the current page self-contained. The canonical Expedition service is
    // the only writer of current-season receipts; its inserts are idempotent.
    await this.expeditions.getGlobal(userId);

    const current = currentExpeditionSeason();
    const rows = await this.prisma.$queryRaw<JournalRow[]>(Prisma.sql`
      WITH recent_seasons AS (
        SELECT receipt.season_key
        FROM dogos_social.expedition_receipts receipt
        WHERE receipt.expedition_key = ${EXPEDITION_KEY}
          AND receipt.expedition_version = ${EXPEDITION_VERSION}
          AND receipt.policy_version = ${EXPEDITION_POLICY_VERSION}
          AND receipt.scope = 'GLOBAL'
          AND receipt.pack_id IS NULL
          AND receipt.user_id = ${userId}
          AND receipt.season_key ~ '^week:[0-9]{4}-[0-9]{2}-[0-9]{2}$'
          AND receipt.season_key <= ${current.key}
        GROUP BY receipt.season_key
        ORDER BY receipt.season_key DESC
        LIMIT ${JOURNAL_MAX_PARTICIPATED_SEASONS}
      )
      SELECT
        receipt.season_key AS "seasonKey",
        receipt.objective_key AS "objectiveKey"
      FROM dogos_social.expedition_receipts receipt
      JOIN recent_seasons season ON season.season_key = receipt.season_key
      WHERE receipt.expedition_key = ${EXPEDITION_KEY}
        AND receipt.expedition_version = ${EXPEDITION_VERSION}
        AND receipt.policy_version = ${EXPEDITION_POLICY_VERSION}
        AND receipt.scope = 'GLOBAL'
        AND receipt.pack_id IS NULL
        AND receipt.user_id = ${userId}
      GROUP BY receipt.season_key, receipt.objective_key
      ORDER BY receipt.season_key DESC, receipt.objective_key ASC
    `);

    const seasons = new Map<string, Set<ExpeditionObjectiveKey>>();
    for (const row of rows) {
      const season = this.parseSeason(row.seasonKey);
      const objective = EXPEDITION_OBJECTIVES.find(
        (candidate) => candidate.key === row.objectiveKey
      );
      if (!season || !objective || season.startsAt > current.startsAt) continue;
      const landmarks = seasons.get(season.key) ?? new Set<ExpeditionObjectiveKey>();
      landmarks.add(objective.key);
      seasons.set(season.key, landmarks);
    }

    const entries = [...seasons.entries()]
      .map(([seasonKey, landmarkKeys]) => {
        const season = this.parseSeason(seasonKey);
        if (!season) return null;
        return {
          season: {
            key: season.key,
            startsAt: season.startsAt.toISOString(),
            endsAt: season.endsAt.toISOString(),
          },
          state: season.key === current.key ? ('ACTIVE' as const) : ('PAST' as const),
          landmarks: EXPEDITION_OBJECTIVES.filter((objective) =>
            landmarkKeys.has(objective.key)
          ).map((objective) => ({ key: objective.key, title: objective.title })),
        };
      })
      .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
      .sort((a, b) => b.season.key.localeCompare(a.season.key));

    return {
      expeditionKey: EXPEDITION_KEY,
      expeditionVersion: EXPEDITION_VERSION,
      policyVersion: EXPEDITION_POLICY_VERSION,
      scope: 'GLOBAL' as const,
      generatedAt: new Date().toISOString(),
      coverage: {
        kind: 'RECENT_PARTICIPATED_SEASONS' as const,
        maxSeasons: JOURNAL_MAX_PARTICIPATED_SEASONS,
      },
      entries,
      principles: [
        'personal-global-receipts-only',
        'landmark-presence-not-volume',
        'no-completion-or-rarity',
        'no-rank-or-streak',
      ],
      disclaimer:
        'Field notes are a bounded memory projection of server-issued Expedition receipts, not an achievement, behavior, health, or pet-performance score.',
    };
  }

  private parseSeason(key: string): JournalSeason | null {
    const match = /^week:(\d{4}-\d{2}-\d{2})$/.exec(key);
    if (!match) return null;

    const startsAt = new Date(`${match[1]}T00:00:00.000Z`);
    if (
      !Number.isFinite(startsAt.getTime()) ||
      startsAt.toISOString().slice(0, 10) !== match[1] ||
      startsAt.getUTCDay() !== 1
    ) {
      return null;
    }

    return {
      key,
      startsAt,
      endsAt: new Date(startsAt.getTime() + 7 * 24 * 60 * 60 * 1000),
    };
  }
}
