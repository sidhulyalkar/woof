import { BadRequestException, Injectable, Logger, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { randomUUID } from 'node:crypto';
import { HouseholdsService } from '../households/households.service';
import { PrismaService } from '../prisma/prisma.service';
import {
  WELLBEING_PATHWAYS,
  type AdventureLearningCareEvent,
  type CanonicalCareEventRecord,
  type CareEventInput,
  type CareSummary,
  type EvidenceType,
  type RewardReceipt,
  type WellbeingPathway,
} from './care-event.types';
import { rewardCareEvent } from './reward-policy';

type EventRow = {
  id: string;
  event_type: string;
  pathway: WellbeingPathway;
  occurred_at: Date;
  outcome: Record<string, unknown> | null;
};

type AdventureLearningEventRow = EventRow & {
  context: Record<string, unknown> | null;
};

type CanonicalEventRow = {
  id: string;
  user_id: string;
  pet_id: string | null;
  event_type: string;
  pathway: WellbeingPathway;
  occurred_at: Date;
  source: string;
  evidence_type: EvidenceType | null;
  evidence_confidence: number;
  context: Record<string, unknown> | null;
  outcome: Record<string, unknown> | null;
  dedupe_key: string;
  visibility: 'PRIVATE' | 'HOUSEHOLD' | 'FRIENDS';
};

type LedgerRow = {
  id: string;
  bond_xp: number;
  policy_version: string;
  explanation: string;
};

type RewardStatsRow = {
  total_xp_today: number;
  pathway_xp_today: number;
  same_pathway_events_today: number;
  repeated_event_count_7d: number;
};

type PathwaySummaryRow = {
  pathway: WellbeingPathway;
  recent_days: number;
  xp: number;
  last_event_at: Date | null;
};

type SelectedQuestContextRow = {
  context: Record<string, unknown> | null;
  created_at: Date;
};

const PATHWAY_LABELS: Record<WellbeingPathway, string> = {
  MOVE: 'Move',
  EXPLORE: 'Explore',
  ENRICH: 'Enrich',
  LEARN: 'Learn',
  CONNECT: 'Connect',
  CARE: 'Care',
  RECOVER: 'Recover',
  BOND: 'Bond',
};

@Injectable()
export class CareEventsService {
  private readonly logger = new Logger(CareEventsService.name);

  constructor(
    private readonly prisma: PrismaService,
    private readonly households: HouseholdsService
  ) {}

  async record(input: CareEventInput): Promise<RewardReceipt> {
    if (input.petId) await this.households.assertPetAccessible(input.userId, input.petId);

    const dedupeScope = input.dedupeScope ?? 'USER';
    if (dedupeScope === 'PET' && !input.petId) {
      throw new BadRequestException('Pet-scoped CareEvent dedupe requires a pet');
    }

    const now = new Date();
    const requestedOccurrenceMs = input.occurredAt?.getTime();
    const occurrenceTimestampNormalized =
      input.occurredAt !== undefined &&
      (!Number.isFinite(requestedOccurrenceMs) || (requestedOccurrenceMs ?? 0) > now.getTime());
    // Offline/historical events retain their original chronology, but an upstream
    // device clock can never place a trusted CareEvent in the future. That prevents
    // future timestamps from inflating Compass/Rhythm recency or ordering.
    const occurredAt = occurrenceTimestampNormalized ? now : (input.occurredAt ?? now);
    const evidenceConfidence = Math.max(0, Math.min(1, input.evidenceConfidence ?? 0.65));
    const visibility = input.visibility ?? 'PRIVATE';

    const receipt = await this.prisma.$transaction(async (tx) => {
      if (dedupeScope === 'PET') {
        // Cross-member Daily Signals retries can arrive under different user IDs and
        // even on different API processes. Serialize the logical pet-scoped identity
        // in PostgreSQL before looking for an existing event. This lock is acquired
        // before the per-user reward lock everywhere PET scope is used, avoiding a
        // reverse lock ordering cycle.
        const petDedupeIdentity = `care-event:pet:${input.petId}:${input.eventType}:${input.dedupeKey}`;
        await tx.$queryRaw<Array<{ acquired: number }>>(Prisma.sql`
          WITH lock_row AS MATERIALIZED (
            SELECT pg_advisory_xact_lock(hashtextextended(${petDedupeIdentity}, 0))
          )
          SELECT 1::int AS acquired FROM lock_row
        `);
      }

      // Serialize reward issuance for one user so concurrent legitimate requests cannot
      // race daily/pathway caps or a user-scoped dedupe key. pg_advisory_xact_lock returns
      // PostgreSQL's `void` pseudo-type, which Prisma cannot deserialize directly, so
      // materialize the lock call and expose only a supported integer scalar.
      await tx.$queryRaw<Array<{ acquired: number }>>(Prisma.sql`
        WITH lock_row AS MATERIALIZED (
          SELECT pg_advisory_xact_lock(hashtextextended(${input.userId}, 0))
        )
        SELECT 1::int AS acquired FROM lock_row
      `);

      const existingPredicate =
        dedupeScope === 'PET'
          ? Prisma.sql`pet_id = ${input.petId} AND event_type = ${input.eventType} AND dedupe_key = ${input.dedupeKey}`
          : Prisma.sql`user_id = ${input.userId} AND dedupe_key = ${input.dedupeKey}`;
      const existing = await tx.$queryRaw<EventRow[]>(Prisma.sql`
        SELECT id, event_type, pathway, occurred_at, outcome
        FROM care_events
        WHERE ${existingPredicate}
        LIMIT 1
      `);

      if (existing[0]) {
        const ledger = await tx.$queryRaw<LedgerRow[]>(Prisma.sql`
          SELECT id, bond_xp, policy_version, explanation
          FROM reward_ledger
          WHERE care_event_id = ${existing[0].id}
          LIMIT 1
        `);
        const prior = ledger[0];
        return {
          careEventId: existing[0].id,
          ledgerId: prior?.id ?? null,
          bondXp: prior?.bond_xp ?? 0,
          pathway: existing[0].pathway,
          policyVersion: prior?.policy_version ?? 'bond-xp-v1',
          explanation: prior?.explanation ?? 'This trusted event was already recorded.',
          duplicate: true,
        };
      }

      // Anti-farming windows are based on trusted server issuance time, not the
      // client-influenced occurrence timestamp. This prevents backdating or future
      // timestamps from manufacturing fresh cap windows. Zero-reward events do not
      // decay later legitimate actions.
      const stats = await tx.$queryRaw<RewardStatsRow[]>(Prisma.sql`
        SELECT
          COALESCE(SUM(CASE WHEN rl.created_at >= date_trunc('day', NOW()) THEN rl.bond_xp ELSE 0 END), 0)::int AS total_xp_today,
          COALESCE(SUM(CASE WHEN rl.created_at >= date_trunc('day', NOW()) AND ce.pathway = ${input.pathway} THEN rl.bond_xp ELSE 0 END), 0)::int AS pathway_xp_today,
          COUNT(*) FILTER (
            WHERE rl.created_at >= date_trunc('day', NOW())
              AND ce.pathway = ${input.pathway}
              AND rl.bond_xp > 0
          )::int AS same_pathway_events_today,
          COUNT(*) FILTER (
            WHERE rl.created_at >= NOW() - INTERVAL '7 days'
              AND ce.event_type = ${input.eventType}
              AND rl.bond_xp > 0
          )::int AS repeated_event_count_7d
        FROM care_events ce
        LEFT JOIN reward_ledger rl ON rl.care_event_id = ce.id
        WHERE ce.user_id = ${input.userId}
      `);

      const reward = rewardCareEvent(input, {
        totalXpToday: stats[0]?.total_xp_today ?? 0,
        pathwayXpToday: stats[0]?.pathway_xp_today ?? 0,
        samePathwayEventsToday: stats[0]?.same_pathway_events_today ?? 0,
        repeatedEventCount7d: stats[0]?.repeated_event_count_7d ?? 0,
      });

      const eventId = randomUUID();
      const ledgerId = randomUUID();
      const contextJson = JSON.stringify(input.context ?? {});
      const outcomeJson = JSON.stringify(input.outcome ?? {});
      const pathwayXpJson = JSON.stringify(reward.pathwayXp);

      await tx.$executeRaw(Prisma.sql`
        INSERT INTO care_events (
          id, user_id, pet_id, event_type, pathway, occurred_at, source,
          evidence_type, evidence_confidence, context, outcome, dedupe_key,
          visibility, created_at
        ) VALUES (
          ${eventId}, ${input.userId}, ${input.petId ?? null}, ${input.eventType},
          ${input.pathway}, ${occurredAt}, ${input.source}, ${input.evidenceType ?? null},
          ${evidenceConfidence}, CAST(${contextJson} AS JSONB), CAST(${outcomeJson} AS JSONB),
          ${input.dedupeKey}, ${visibility}, NOW()
        )
      `);

      await tx.$executeRaw(Prisma.sql`
        INSERT INTO reward_ledger (
          id, care_event_id, user_id, pet_id, bond_xp, pathway_xp,
          policy_version, explanation, created_at
        ) VALUES (
          ${ledgerId}, ${eventId}, ${input.userId}, ${input.petId ?? null},
          ${reward.bondXp}, CAST(${pathwayXpJson} AS JSONB), ${reward.policyVersion},
          ${reward.explanation}, NOW()
        )
      `);

      if (reward.bondXp > 0) {
        // Keep the legacy aggregate synchronized for older surfaces while all new
        // reward reads come from the immutable ledger.
        await tx.user.update({
          where: { id: input.userId },
          data: { totalPoints: { increment: reward.bondXp } },
        });
      }

      return {
        careEventId: eventId,
        ledgerId,
        bondXp: reward.bondXp,
        pathway: input.pathway,
        policyVersion: reward.policyVersion,
        explanation: reward.explanation,
        duplicate: false,
      };
    });

    // Keep reward observability useful without placing user/pet identifiers in logs.
    this.logger.log(
      JSON.stringify({
        event: 'adventure_reward_decision',
        careEventId: receipt.careEventId,
        eventType: input.eventType,
        pathway: receipt.pathway,
        bondXp: receipt.bondXp,
        duplicate: receipt.duplicate,
        dedupeScope,
        policyVersion: receipt.policyVersion,
        occurrenceTimestampNormalized,
      })
    );

    return receipt;
  }

  async getAuthorizedEvent(userId: string, eventId: string): Promise<CanonicalCareEventRecord> {
    const rows = await this.prisma.$queryRaw<CanonicalEventRow[]>(Prisma.sql`
      SELECT
        id,
        user_id,
        pet_id,
        event_type,
        pathway,
        occurred_at,
        source,
        evidence_type,
        evidence_confidence,
        context,
        outcome,
        dedupe_key,
        visibility
      FROM care_events
      WHERE id = ${eventId}
      LIMIT 1
    `);
    const row = rows[0];
    if (!row) throw new NotFoundException('CareEvent not found');

    if (row.pet_id) {
      await this.households.assertPetAccessible(userId, row.pet_id);
    } else if (row.user_id !== userId) {
      throw new NotFoundException('CareEvent not found');
    }

    return {
      id: row.id,
      userId: row.user_id,
      petId: row.pet_id,
      eventType: row.event_type,
      pathway: row.pathway,
      occurredAt: row.occurred_at.toISOString(),
      source: row.source,
      evidenceType: row.evidence_type,
      evidenceConfidence: row.evidence_confidence,
      context: row.context ?? {},
      outcome: row.outcome ?? {},
      dedupeKey: row.dedupe_key,
      visibility: row.visibility,
    };
  }

  async recordQuestInteraction(input: {
    userId: string;
    petId: string;
    questId: string;
    interaction: 'SHOWN' | 'SELECTED' | 'DISMISSED' | 'COMPLETED';
    pathway: WellbeingPathway;
    context?: Record<string, unknown>;
  }) {
    await this.households.assertPetAccessible(input.userId, input.petId);
    const id = randomUUID();
    const rows = await this.prisma.$queryRaw<Array<{ id: string }>>(Prisma.sql`
      INSERT INTO quest_interactions (
        id, user_id, pet_id, quest_id, interaction, pathway, context, created_at
      ) VALUES (
        ${id}, ${input.userId}, ${input.petId}, ${input.questId}, ${input.interaction},
        ${input.pathway}, CAST(${JSON.stringify(input.context ?? {})} AS JSONB), NOW()
      )
      ON CONFLICT (user_id, pet_id, quest_id, interaction)
      DO UPDATE SET
        pathway = EXCLUDED.pathway,
        context = EXCLUDED.context,
        created_at = EXCLUDED.created_at
      RETURNING id
    `);
    return { id: rows[0]?.id ?? id };
  }

  async getRecentSelectedQuestContext(userId: string, petId: string, questId: string) {
    await this.households.assertPetAccessible(userId, petId);
    const rows = await this.prisma.$queryRaw<SelectedQuestContextRow[]>(Prisma.sql`
      SELECT context, created_at
      FROM quest_interactions
      WHERE user_id = ${userId}
        AND pet_id = ${petId}
        AND quest_id = ${questId}
        AND interaction = 'SELECTED'
        AND created_at >= NOW() - INTERVAL '72 hours'
      ORDER BY created_at DESC
      LIMIT 1
    `);
    return rows[0] ?? null;
  }

  async getAdventureLearningEvents(
    userId: string,
    petId: string,
    limit = 24
  ): Promise<AdventureLearningCareEvent[]> {
    await this.households.assertPetAccessible(userId, petId);
    const boundedLimit = Math.max(1, Math.min(24, Math.round(limit)));
    const rows = await this.prisma.$queryRaw<AdventureLearningEventRow[]>(Prisma.sql`
      SELECT id, event_type, pathway, occurred_at, context, outcome
      FROM care_events
      WHERE user_id = ${userId}
        AND pet_id = ${petId}
        AND source = 'QUEST_ENGINE'
        AND (event_type = 'SAFE_OPT_OUT' OR LEFT(event_type, 6) = 'QUEST_')
        AND occurred_at >= NOW() - INTERVAL '28 days'
      ORDER BY occurred_at DESC, id ASC
      LIMIT ${boundedLimit}
    `);

    return rows.map((row) => ({
      id: row.id,
      eventType: row.event_type,
      pathway: row.pathway,
      occurredAt: row.occurred_at.toISOString(),
      context: row.context,
      outcome: row.outcome,
    }));
  }

  async getSummary(userId: string, petId?: string): Promise<CareSummary> {
    if (petId) await this.households.assertPetAccessible(userId, petId);

    const petFilter = petId ? Prisma.sql`AND ce.pet_id = ${petId}` : Prisma.empty;
    const [totalRows, pathwayRows, recentRows, rhythmRows] = await Promise.all([
      this.prisma.$queryRaw<Array<{ bond_xp: number }>>(Prisma.sql`
        SELECT COALESCE(SUM(rl.bond_xp), 0)::int AS bond_xp
        FROM reward_ledger rl
        JOIN care_events ce ON ce.id = rl.care_event_id
        WHERE ce.user_id = ${userId} ${petFilter}
      `),
      this.prisma.$queryRaw<PathwaySummaryRow[]>(Prisma.sql`
        SELECT
          ce.pathway,
          COUNT(DISTINCT DATE(ce.occurred_at)) FILTER (WHERE ce.occurred_at >= NOW() - INTERVAL '28 days')::int AS recent_days,
          COALESCE(SUM(rl.bond_xp), 0)::int AS xp,
          MAX(ce.occurred_at) AS last_event_at
        FROM care_events ce
        LEFT JOIN reward_ledger rl ON rl.care_event_id = ce.id
        WHERE ce.user_id = ${userId} ${petFilter}
        GROUP BY ce.pathway
      `),
      this.prisma.$queryRaw<Array<EventRow & { bond_xp: number }>>(Prisma.sql`
        SELECT ce.id, ce.event_type, ce.pathway, ce.occurred_at, ce.outcome,
               COALESCE(rl.bond_xp, 0)::int AS bond_xp
        FROM care_events ce
        LEFT JOIN reward_ledger rl ON rl.care_event_id = ce.id
        WHERE ce.user_id = ${userId} ${petFilter}
        ORDER BY ce.occurred_at DESC
        LIMIT 12
      `),
      this.prisma.$queryRaw<Array<{ active_weeks: number }>>(Prisma.sql`
        SELECT COUNT(DISTINCT date_trunc('week', ce.occurred_at))::int AS active_weeks
        FROM care_events ce
        WHERE ce.user_id = ${userId}
          AND ce.occurred_at >= NOW() - INTERVAL '35 days'
          ${petFilter}
      `),
    ]);

    const byPathway = new Map(pathwayRows.map((row) => [row.pathway, row]));
    const activeWeeks = Math.min(5, rhythmRows[0]?.active_weeks ?? 0);

    return {
      bondXp: totalRows[0]?.bond_xp ?? 0,
      rhythm: {
        activeWeeks,
        windowWeeks: 5,
        label:
          activeWeeks >= 4
            ? 'Great rhythm this month'
            : activeWeeks >= 2
              ? 'A steady rhythm is growing'
              : 'Every useful moment can start a rhythm',
      },
      pathways: WELLBEING_PATHWAYS.map((pathway) => {
        const row = byPathway.get(pathway);
        const recentDays = row?.recent_days ?? 0;
        return {
          pathway,
          label: PATHWAY_LABELS[pathway],
          recentDays,
          // This is opportunity coverage, not a health score. Four distinct recent
          // days fills the visual treatment without implying a universal prescription.
          coverage: Math.min(100, recentDays * 25),
          xp: row?.xp ?? 0,
          lastEventAt: row?.last_event_at?.toISOString() ?? null,
        };
      }),
      recentEvents: recentRows.map((row) => ({
        id: row.id,
        eventType: row.event_type,
        pathway: row.pathway,
        occurredAt: row.occurred_at.toISOString(),
        outcome: row.outcome,
        bondXp: row.bond_xp,
      })),
    };
  }
}
