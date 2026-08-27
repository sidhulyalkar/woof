import { Injectable } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';

type CandidateRow = {
  sourceId: string;
  petId: string | null;
  petName: string | null;
  pathway: string;
  occurredAt: Date;
  context: unknown;
  outcome: unknown;
};

@Injectable()
export class SocialAdventureShareCandidatesService {
  constructor(private readonly prisma: PrismaService) {}

  async list(userId: string) {
    const rows = await this.prisma.$queryRaw<CandidateRow[]>(Prisma.sql`
      SELECT
        event.id AS "sourceId",
        event.pet_id AS "petId",
        pet.name AS "petName",
        event.pathway,
        event.occurred_at AS "occurredAt",
        event.context,
        event.outcome
      FROM public.care_events event
      LEFT JOIN public.pets pet ON pet.id = event.pet_id
      WHERE event.user_id = ${userId}
        AND event.source = 'QUEST_ENGINE'
        AND NOT EXISTS (
          SELECT 1
          FROM dogos_social.shares share
          WHERE share.user_id = ${userId}
            AND share.source_type = 'CARE_EVENT'
            AND share.source_id = event.id
        )
      ORDER BY event.occurred_at DESC, event.id DESC
      LIMIT 6
    `);

    return {
      candidates: rows.map((row) => {
        const context = this.asRecord(row.context);
        const outcome = this.asRecord(row.outcome);
        const safeOptOut = outcome.safeOptOut === true;
        const dogExperience =
          typeof outcome.dogExperience === 'string' ? outcome.dogExperience : null;
        const petName = row.petName ?? 'your dog';
        const kind = safeOptOut
          ? 'GOOD_READ'
          : dogExperience === 'not_their_thing'
            ? 'DISCOVERY'
            : 'ADVENTURE_MEMORY';
        const headline =
          typeof context.questTitle === 'string' && context.questTitle.trim()
            ? context.questTitle.trim().slice(0, 100)
            : 'Shared Adventure';
        const summary = safeOptOut
          ? `You listened to ${petName} and changed course. That is shareable as a good read, not a failed quest.`
          : kind === 'DISCOVERY'
            ? `You learned something useful about what fits ${petName}.`
            : `A ${row.pathway.toLowerCase()} moment with ${petName}.`;

        return {
          sourceType: 'CARE_EVENT' as const,
          sourceId: row.sourceId,
          petId: row.petId,
          petName: row.petName,
          kind,
          headline,
          summary,
          occurredAt: row.occurredAt.toISOString(),
        };
      }),
      privacy:
        'These are private previews. Nothing enters the social feed until the owner explicitly shares one.',
    };
  }

  private asRecord(value: unknown): Record<string, unknown> {
    return value && typeof value === 'object' && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : {};
  }
}
