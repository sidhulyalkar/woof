import { Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { randomUUID } from 'node:crypto';
import { PrismaService } from '../prisma/prisma.service';
import {
  WELLBEING_PATHWAYS,
  type CareEventInput,
  type CareSummary,
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
  constructor(private readonly prisma: PrismaService) {}

  async record(input: CareEventInput): Promise<RewardReceipt> {
    if (input.petId) await this.assertOwnedPet(input.userId, input.petId);

    const occurredAt = input.occurredAt ?? new Date();
    const evidenceConfidence = Math.max(0, Math.min(1, input.evidenceConfidence ?? 0.65));
    const visibility = input.visibility ?? 'PRIVATE';

    return this.prisma.$transaction(async (tx) => {
      // Serialize reward issuance for one user so concurrent legitimate requests cannot
      // race the daily/pathway caps or a shared dedupe key. The lock lives only for
      // this transaction and does not block rewards for other users.
      await tx.$queryRaw(
        Prisma.sql`SELECT pg_advisory_xact_lock(hashtextextended(${input.userId}, 0))`
      );

      const existing = await tx.$queryRaw<EventRow[]>(Prisma.sql`
        SELECT id, event_type, pathway, occurred_at, outcome
        FROM care_events
        WHERE user_id = ${input.userId} AND dedupe_key = ${input.dedupeKey}
        LIMIT 1
      `);

      if (existing[0]) {
        const ledger = await tx.$queryRaw<LedgerRow[]>(Prisma.sql`
          SELECT id, bond_xp, policy_version, explanation
          FROM reward_ledger
          WHERE care_event_id = ${existing[0].id}
          LIMIT 1
        `);
        const receipt = ledger[0];
        return {
          careEventId: existing[0].id,
          ledgerId: receipt?.id ?? null,
          bondXp: receipt?.bond_xp ?? 0,
          pathway: existing[0].pathway,
          policyVersion: receipt?.policy_version ?? 'bond-xp-v1',
          explanation: receipt?.explanation ?? 'This trusted event was already recorded.',
          duplicate: true,
        };
      }

      const stats = await tx.$queryRaw<RewardStatsRow[]>(Prisma.sql`
        SELECT
          COALESCE(SUM(CASE WHEN ce.occurred_at >= date_trunc('day', ${occurredAt}::timestamp) THEN rl.bond_xp ELSE 0 END), 0)::int AS total_xp_today,
          COALESCE(SUM(CASE WHEN ce.occurred_at >= date_trunc('day', ${occurredAt}::timestamp) AND ce.pathway = ${input.pathway} THEN rl.bond_xp ELSE 0 END), 0)::int AS pathway_xp_today,
          COUNT(*) FILTER (WHERE ce.occurred_at >= date_trunc('day', ${occurredAt}::timestamp) AND ce.pathway = ${input.pathway})::int AS same_pathway_events_today,
          COUNT(*) FILTER (WHERE ce.occurred_at >= ${new Date(occurredAt.getTime() - 7 * 24 * 60 * 60 * 1000)} AND ce.event_type = ${input.eventType})::int AS repeated_event_count_7d
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
  }

  async recordQuestInteraction(input: {
    userId: string;
    petId: string;
    questId: string;
    interaction: 'SHOWN' | 'SELECTED' | 'DISMISSED' | 'COMPLETED';
    pathway: WellbeingPathway;
    context?: Record<string, unknown>;
  }) {
    await this.assertOwnedPet(input.userId, input.petId);
    const id = randomUUID();
    await this.prisma.$executeRaw(Prisma.sql`
      INSERT INTO quest_interactions (
        id, user_id, pet_id, quest_id, interaction, pathway, context, created_at
      ) VALUES (
        ${id}, ${input.userId}, ${input.petId}, ${input.questId}, ${input.interaction},
        ${input.pathway}, CAST(${JSON.stringify(input.context ?? {})} AS JSONB), NOW()
      )
    `);
    return { id };
  }

  async getSummary(userId: string, petId?: string): Promise<CareSummary> {
    if (petId) await this.assertOwnedPet(userId, petId);

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

  private async assertOwnedPet(userId: string, petId: string) {
    const pet = await this.prisma.pet.findFirst({
      where: { id: petId, ownerId: userId },
      select: { id: true },
    });
    if (!pet) throw new NotFoundException('Pet not found');
    return pet;
  }
}
