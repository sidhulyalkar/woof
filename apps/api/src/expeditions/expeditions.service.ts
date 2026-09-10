import { Injectable } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { PackAccessService, type PackAccess } from '../social-adventure/pack-access.service';
import {
  currentExpeditionSeason,
  EXPEDITION_KEY,
  EXPEDITION_OBJECTIVES,
  EXPEDITION_POLICY_VERSION,
  EXPEDITION_VERSION,
  type ExpeditionScope,
} from './expeditions.policy';

type ProjectionRow = {
  objectiveKey: string;
  total: number;
  contributors: number;
  mine: number;
};

type ExpeditionSeason = ReturnType<typeof currentExpeditionSeason>;

@Injectable()
export class ExpeditionsService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly packAccess: PackAccessService
  ) {}

  async getGlobal(userId: string) {
    const season = currentExpeditionSeason();
    await this.materializeGlobalReceipts(season);
    const rows = await this.readProjection('GLOBAL', null, userId, season);
    return this.response('GLOBAL', null, null, rows, season);
  }

  async getPack(userId: string, packId: string) {
    const pack = await this.packAccess.requireActiveMembership(userId, packId);
    const season = currentExpeditionSeason();
    await this.materializePackReceipts(packId, season);
    const rows = await this.readProjection('PACK', packId, userId, season);
    return this.response('PACK', packId, pack, rows, season);
  }

  async getLegacyGlobalChallenges(userId: string) {
    const projection = await this.getGlobal(userId);
    const challenges = EXPEDITION_OBJECTIVES.filter(
      (objective) => objective.legacyChallengeId && objective.legacyTarget && objective.legacyUnit
    ).map((objective) => {
      const projected = projection.objectives.find((item) => item.key === objective.key);
      const total = projected?.total ?? 0;
      const target = objective.legacyTarget ?? 1;
      return {
        id: objective.legacyChallengeId,
        title: objective.legacyTitle,
        description: objective.description,
        pathways: [...objective.categories],
        target,
        unit: objective.legacyUnit,
        total,
        contributors: projected?.contributors ?? 0,
        myContribution: projected?.myContribution ?? 0,
        progress: Math.min(1, total / target),
        completed: total >= target,
      };
    });

    return {
      generatedAt: projection.generatedAt,
      windowDays: 7,
      challenges,
      principles: [
        'legacy-global-compatibility',
        'everyone-contributes',
        'nobody-loses',
        'canonical-adventures-only',
        'bounded-per-human-contribution',
        'no-raw-distance-ranking',
        'no-medical-competition',
      ],
      authority: {
        endpoint: '/expeditions/global',
        expeditionKey: projection.expeditionKey,
        expeditionVersion: projection.expeditionVersion,
        seasonKey: projection.season.key,
        policyVersion: projection.policyVersion,
      },
    };
  }

  private async materializeGlobalReceipts(season: ExpeditionSeason) {
    await Promise.all([
      this.materializeGlobalCareReceipts(season),
      this.materializeGlobalHumanSkillReceipts(season),
    ]);
  }

  private async materializePackReceipts(packId: string, season: ExpeditionSeason) {
    // Membership is checked before this method is called for the viewer. Each
    // contributor is independently constrained to a current ACTIVE membership,
    // and evidence from before the current joined_at never becomes Pack progress.
    await Promise.all([
      this.materializePackCareReceipts(packId, season),
      this.materializePackHumanSkillReceipts(packId, season),
    ]);
  }

  private async materializeGlobalCareReceipts(season: ExpeditionSeason) {
    await this.prisma.$executeRaw(Prisma.sql`
    WITH issued AS (
      SELECT
        receipt.user_id,
        receipt.category_key,
        COUNT(*)::int AS issued_count
      FROM dogos_social.expedition_receipts receipt
      WHERE receipt.expedition_key = ${EXPEDITION_KEY}
        AND receipt.expedition_version = ${EXPEDITION_VERSION}
        AND receipt.season_key = ${season.key}
        AND receipt.policy_version = ${EXPEDITION_POLICY_VERSION}
        AND receipt.scope = 'GLOBAL'
        AND receipt.pack_id IS NULL
        AND receipt.source_type = 'CARE_EVENT'
      GROUP BY receipt.user_id, receipt.category_key
    ), ranked AS (
      SELECT
        event.id,
        event.user_id,
        event.pathway,
        event.occurred_at,
        COALESCE(issued.issued_count, 0) AS issued_count,
        ROW_NUMBER() OVER (
          PARTITION BY event.user_id, event.pathway
          ORDER BY event.occurred_at ASC, event.id ASC
        ) AS category_rank
      FROM public.care_events event
      LEFT JOIN issued
        ON issued.user_id = event.user_id
       AND issued.category_key = event.pathway
      WHERE event.occurred_at >= ${season.startsAt}
        AND event.occurred_at < ${season.endsAt}
        AND event.source = 'QUEST_ENGINE'
        AND event.event_type LIKE 'QUEST_%'
        AND event.pathway IN ('EXPLORE', 'ENRICH', 'RECOVER')
        AND NOT EXISTS (
          SELECT 1
          FROM dogos_social.expedition_receipts existing
          WHERE existing.expedition_key = ${EXPEDITION_KEY}
            AND existing.expedition_version = ${EXPEDITION_VERSION}
            AND existing.season_key = ${season.key}
            AND existing.policy_version = ${EXPEDITION_POLICY_VERSION}
            AND existing.scope = 'GLOBAL'
            AND existing.pack_id IS NULL
            AND existing.source_type = 'CARE_EVENT'
            AND existing.source_id = event.id
        )
    ), eligible AS (
      SELECT
        id,
        user_id,
        pathway,
        occurred_at,
        CASE
          WHEN pathway IN ('EXPLORE', 'ENRICH') THEN 'SNIFF_EXPLORE'
          ELSE 'RECOVERY_COUNTS'
        END AS objective_key
      FROM ranked
      WHERE category_rank + issued_count <= 2
    ), prepared AS (
      SELECT
        *,
        concat_ws(
          '|',
          ${EXPEDITION_KEY},
          ${EXPEDITION_VERSION},
          ${season.key},
          ${EXPEDITION_POLICY_VERSION},
          'GLOBAL',
          objective_key,
          'CARE_EVENT',
          id
        ) AS identity
      FROM eligible
    ), fingerprinted AS (
      SELECT
        *,
        md5(identity) || md5('expedition-v1|' || identity) AS fingerprint
      FROM prepared
    )
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
    )
    SELECT
      'exp:' || fingerprint,
      ${EXPEDITION_KEY},
      ${EXPEDITION_VERSION},
      ${season.key},
      ${EXPEDITION_POLICY_VERSION},
      'GLOBAL',
      NULL,
      user_id,
      'CARE_EVENT',
      id,
      objective_key,
      pathway,
      pathway,
      fingerprint,
      occurred_at
    FROM fingerprinted
    ON CONFLICT DO NOTHING
  `);
  }

  private async materializeGlobalHumanSkillReceipts(season: ExpeditionSeason) {
    await this.prisma.$executeRaw(Prisma.sql`
    WITH issued AS (
      SELECT
        receipt.user_id,
        receipt.category_key,
        COUNT(*)::int AS issued_count
      FROM dogos_social.expedition_receipts receipt
      WHERE receipt.expedition_key = ${EXPEDITION_KEY}
        AND receipt.expedition_version = ${EXPEDITION_VERSION}
        AND receipt.season_key = ${season.key}
        AND receipt.policy_version = ${EXPEDITION_POLICY_VERSION}
        AND receipt.scope = 'GLOBAL'
        AND receipt.pack_id IS NULL
        AND receipt.source_type = 'HUMAN_SKILL_ATTEMPT'
      GROUP BY receipt.user_id, receipt.category_key
    ), ranked AS (
      SELECT
        attempt.id,
        attempt.user_id,
        attempt.challenge_key,
        attempt.completed_at,
        COALESCE(issued.issued_count, 0) AS issued_count,
        ROW_NUMBER() OVER (
          PARTITION BY attempt.user_id, attempt.challenge_key
          ORDER BY attempt.completed_at ASC, attempt.id ASC
        ) AS category_rank
      FROM dogos_social.human_skill_attempts attempt
      LEFT JOIN issued
        ON issued.user_id = attempt.user_id
       AND issued.category_key = attempt.challenge_key
      WHERE attempt.completed_at >= ${season.startsAt}
        AND attempt.completed_at < ${season.endsAt}
        AND attempt.challenge_key IN (
          'MAKE_IT_EASIER',
          'CATCH_THE_GOOD',
          'PAIRING_LAB',
          'MARKER_TIMING'
        )
        AND NOT EXISTS (
          SELECT 1
          FROM dogos_social.expedition_receipts existing
          WHERE existing.expedition_key = ${EXPEDITION_KEY}
            AND existing.expedition_version = ${EXPEDITION_VERSION}
            AND existing.season_key = ${season.key}
            AND existing.policy_version = ${EXPEDITION_POLICY_VERSION}
            AND existing.scope = 'GLOBAL'
            AND existing.pack_id IS NULL
            AND existing.source_type = 'HUMAN_SKILL_ATTEMPT'
            AND existing.source_id = attempt.id
        )
    ), eligible AS (
      SELECT * FROM ranked WHERE category_rank + issued_count <= 1
    ), prepared AS (
      SELECT
        *,
        concat_ws(
          '|',
          ${EXPEDITION_KEY},
          ${EXPEDITION_VERSION},
          ${season.key},
          ${EXPEDITION_POLICY_VERSION},
          'GLOBAL',
          'READ_THE_ROOM',
          'HUMAN_SKILL_ATTEMPT',
          id
        ) AS identity
      FROM eligible
    ), fingerprinted AS (
      SELECT
        *,
        md5(identity) || md5('expedition-v1|' || identity) AS fingerprint
      FROM prepared
    )
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
    )
    SELECT
      'exp:' || fingerprint,
      ${EXPEDITION_KEY},
      ${EXPEDITION_VERSION},
      ${season.key},
      ${EXPEDITION_POLICY_VERSION},
      'GLOBAL',
      NULL,
      user_id,
      'HUMAN_SKILL_ATTEMPT',
      id,
      'READ_THE_ROOM',
      challenge_key,
      NULL,
      fingerprint,
      completed_at
    FROM fingerprinted
    ON CONFLICT DO NOTHING
  `);
  }

  private async materializePackCareReceipts(packId: string, season: ExpeditionSeason) {
    await this.prisma.$executeRaw(Prisma.sql`
    WITH issued AS (
      SELECT
        receipt.user_id,
        receipt.category_key,
        COUNT(*)::int AS issued_count
      FROM dogos_social.expedition_receipts receipt
      WHERE receipt.expedition_key = ${EXPEDITION_KEY}
        AND receipt.expedition_version = ${EXPEDITION_VERSION}
        AND receipt.season_key = ${season.key}
        AND receipt.policy_version = ${EXPEDITION_POLICY_VERSION}
        AND receipt.scope = 'PACK'
        AND receipt.pack_id = ${packId}
        AND receipt.source_type = 'CARE_EVENT'
      GROUP BY receipt.user_id, receipt.category_key
    ), ranked AS (
      SELECT
        event.id,
        event.user_id,
        event.pathway,
        event.occurred_at,
        COALESCE(issued.issued_count, 0) AS issued_count,
        ROW_NUMBER() OVER (
          PARTITION BY event.user_id, event.pathway
          ORDER BY event.occurred_at ASC, event.id ASC
        ) AS category_rank
      FROM public.care_events event
      JOIN dogos_social.pack_memberships member
        ON member.user_id = event.user_id
       AND member.pack_id = ${packId}
       AND member.status = 'ACTIVE'
      LEFT JOIN issued
        ON issued.user_id = event.user_id
       AND issued.category_key = event.pathway
      WHERE event.occurred_at >= GREATEST(${season.startsAt}, member.joined_at)
        AND event.occurred_at < ${season.endsAt}
        AND event.source = 'QUEST_ENGINE'
        AND event.event_type LIKE 'QUEST_%'
        AND event.pathway IN ('EXPLORE', 'ENRICH', 'RECOVER')
        AND NOT EXISTS (
          SELECT 1
          FROM dogos_social.expedition_receipts existing
          WHERE existing.expedition_key = ${EXPEDITION_KEY}
            AND existing.expedition_version = ${EXPEDITION_VERSION}
            AND existing.season_key = ${season.key}
            AND existing.policy_version = ${EXPEDITION_POLICY_VERSION}
            AND existing.scope = 'PACK'
            AND existing.pack_id = ${packId}
            AND existing.source_type = 'CARE_EVENT'
            AND existing.source_id = event.id
        )
    ), eligible AS (
      SELECT
        id,
        user_id,
        pathway,
        occurred_at,
        CASE
          WHEN pathway IN ('EXPLORE', 'ENRICH') THEN 'SNIFF_EXPLORE'
          ELSE 'RECOVERY_COUNTS'
        END AS objective_key
      FROM ranked
      WHERE category_rank + issued_count <= 2
    ), prepared AS (
      SELECT
        *,
        concat_ws(
          '|',
          ${EXPEDITION_KEY},
          ${EXPEDITION_VERSION},
          ${season.key},
          ${EXPEDITION_POLICY_VERSION},
          'PACK',
          ${packId},
          objective_key,
          'CARE_EVENT',
          id
        ) AS identity
      FROM eligible
    ), fingerprinted AS (
      SELECT
        *,
        md5(identity) || md5('expedition-v1|' || identity) AS fingerprint
      FROM prepared
    )
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
    )
    SELECT
      'exp:' || fingerprint,
      ${EXPEDITION_KEY},
      ${EXPEDITION_VERSION},
      ${season.key},
      ${EXPEDITION_POLICY_VERSION},
      'PACK',
      ${packId},
      user_id,
      'CARE_EVENT',
      id,
      objective_key,
      pathway,
      pathway,
      fingerprint,
      occurred_at
    FROM fingerprinted
    ON CONFLICT DO NOTHING
  `);
  }

  private async materializePackHumanSkillReceipts(packId: string, season: ExpeditionSeason) {
    await this.prisma.$executeRaw(Prisma.sql`
    WITH issued AS (
      SELECT
        receipt.user_id,
        receipt.category_key,
        COUNT(*)::int AS issued_count
      FROM dogos_social.expedition_receipts receipt
      WHERE receipt.expedition_key = ${EXPEDITION_KEY}
        AND receipt.expedition_version = ${EXPEDITION_VERSION}
        AND receipt.season_key = ${season.key}
        AND receipt.policy_version = ${EXPEDITION_POLICY_VERSION}
        AND receipt.scope = 'PACK'
        AND receipt.pack_id = ${packId}
        AND receipt.source_type = 'HUMAN_SKILL_ATTEMPT'
      GROUP BY receipt.user_id, receipt.category_key
    ), ranked AS (
      SELECT
        attempt.id,
        attempt.user_id,
        attempt.challenge_key,
        attempt.completed_at,
        COALESCE(issued.issued_count, 0) AS issued_count,
        ROW_NUMBER() OVER (
          PARTITION BY attempt.user_id, attempt.challenge_key
          ORDER BY attempt.completed_at ASC, attempt.id ASC
        ) AS category_rank
      FROM dogos_social.human_skill_attempts attempt
      JOIN dogos_social.pack_memberships member
        ON member.user_id = attempt.user_id
       AND member.pack_id = ${packId}
       AND member.status = 'ACTIVE'
      LEFT JOIN issued
        ON issued.user_id = attempt.user_id
       AND issued.category_key = attempt.challenge_key
      WHERE attempt.completed_at >= GREATEST(${season.startsAt}, member.joined_at)
        AND attempt.completed_at < ${season.endsAt}
        AND attempt.challenge_key IN (
          'MAKE_IT_EASIER',
          'CATCH_THE_GOOD',
          'PAIRING_LAB',
          'MARKER_TIMING'
        )
        AND NOT EXISTS (
          SELECT 1
          FROM dogos_social.expedition_receipts existing
          WHERE existing.expedition_key = ${EXPEDITION_KEY}
            AND existing.expedition_version = ${EXPEDITION_VERSION}
            AND existing.season_key = ${season.key}
            AND existing.policy_version = ${EXPEDITION_POLICY_VERSION}
            AND existing.scope = 'PACK'
            AND existing.pack_id = ${packId}
            AND existing.source_type = 'HUMAN_SKILL_ATTEMPT'
            AND existing.source_id = attempt.id
        )
    ), eligible AS (
      SELECT * FROM ranked WHERE category_rank + issued_count <= 1
    ), prepared AS (
      SELECT
        *,
        concat_ws(
          '|',
          ${EXPEDITION_KEY},
          ${EXPEDITION_VERSION},
          ${season.key},
          ${EXPEDITION_POLICY_VERSION},
          'PACK',
          ${packId},
          'READ_THE_ROOM',
          'HUMAN_SKILL_ATTEMPT',
          id
        ) AS identity
      FROM eligible
    ), fingerprinted AS (
      SELECT
        *,
        md5(identity) || md5('expedition-v1|' || identity) AS fingerprint
      FROM prepared
    )
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
    )
    SELECT
      'exp:' || fingerprint,
      ${EXPEDITION_KEY},
      ${EXPEDITION_VERSION},
      ${season.key},
      ${EXPEDITION_POLICY_VERSION},
      'PACK',
      ${packId},
      user_id,
      'HUMAN_SKILL_ATTEMPT',
      id,
      'READ_THE_ROOM',
      challenge_key,
      NULL,
      fingerprint,
      completed_at
    FROM fingerprinted
    ON CONFLICT DO NOTHING
  `);
  }

  private async readProjection(
    scope: ExpeditionScope,
    packId: string | null,
    userId: string,
    season: ExpeditionSeason
  ) {
    const scopePredicate =
      scope === 'GLOBAL'
        ? Prisma.sql`receipt.scope = 'GLOBAL' AND receipt.pack_id IS NULL`
        : Prisma.sql`receipt.scope = 'PACK' AND receipt.pack_id = ${packId}`;

    return this.prisma.$queryRaw<ProjectionRow[]>(Prisma.sql`
      SELECT
        receipt.objective_key AS "objectiveKey",
        COUNT(*)::int AS total,
        COUNT(DISTINCT receipt.user_id)::int AS contributors,
        COUNT(*) FILTER (WHERE receipt.user_id = ${userId})::int AS mine
      FROM dogos_social.expedition_receipts receipt
      WHERE receipt.expedition_key = ${EXPEDITION_KEY}
        AND receipt.expedition_version = ${EXPEDITION_VERSION}
        AND receipt.season_key = ${season.key}
        AND receipt.policy_version = ${EXPEDITION_POLICY_VERSION}
        AND ${scopePredicate}
      GROUP BY receipt.objective_key
    `);
  }

  private response(
    scope: ExpeditionScope,
    packId: string | null,
    pack: PackAccess | null,
    rows: ProjectionRow[],
    season: ExpeditionSeason
  ) {
    return {
      expeditionKey: EXPEDITION_KEY,
      expeditionVersion: EXPEDITION_VERSION,
      policyVersion: EXPEDITION_POLICY_VERSION,
      scope,
      ...(scope === 'PACK' && pack
        ? { pack: { id: packId, name: pack.name, memberCount: pack.memberCount } }
        : {}),
      season: {
        key: season.key,
        startsAt: season.startsAt.toISOString(),
        endsAt: season.endsAt.toISOString(),
      },
      generatedAt: new Date().toISOString(),
      objectives: EXPEDITION_OBJECTIVES.map((objective) => {
        const row = rows.find((candidate) => candidate.objectiveKey === objective.key);
        return {
          key: objective.key,
          title: objective.title,
          description: objective.description,
          sourceType: objective.sourceType,
          categories: [...objective.categories],
          total: row?.total ?? 0,
          contributors: row?.contributors ?? 0,
          myContribution: row?.mine ?? 0,
          cap: {
            perContributor: objective.perContributorCap,
            perCategory: objective.perCategoryCap,
          },
          target: null,
          status: 'CALIBRATING' as const,
        };
      }),
      principles: [
        'canonical-evidence-only',
        'bounded-per-human-contribution',
        'breadth-over-volume',
        'active-membership-for-pack-scope',
        'no-care-or-medical-competition',
        'no-distance-duration-or-intensity-points',
        'no-popularity-or-streak-points',
        'human-skill-score-magnitude-does-not-count',
      ],
      calibration:
        'Season completion targets are intentionally unset in authority v1. Pilot evidence should calibrate cooperative targets from bounded participation breadth, not raw activity volume.',
    };
  }
}
