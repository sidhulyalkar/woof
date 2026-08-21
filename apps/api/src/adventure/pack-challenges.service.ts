import { Injectable } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';

type AggregateRow = {
  total: number;
  contributors: number;
  mine: number;
};

type ChallengeDefinition = {
  id: string;
  title: string;
  description: string;
  pathways: string[];
  target: number;
  unit: string;
};

const CHALLENGES: ChallengeDefinition[] = [
  {
    id: 'sniff-explore-week',
    title: 'Sniff & Explore Week',
    description:
      'Together, make room for exploration and enrichment. Every useful session contributes once.',
    pathways: ['EXPLORE', 'ENRICH'],
    target: 250,
    unit: 'shared adventures',
  },
  {
    id: 'recovery-counts-week',
    title: 'Recovery Counts',
    description: 'Make rest and decompression visible as legitimate care, not a broken streak.',
    pathways: ['RECOVER'],
    target: 100,
    unit: 'recovery moments',
  },
];

@Injectable()
export class PackChallengesService {
  constructor(private readonly prisma: PrismaService) {}

  async getChallenges(userId: string) {
    const challenges = await Promise.all(
      CHALLENGES.map(async (challenge) => {
        const rows = await this.prisma.$queryRaw<AggregateRow[]>(Prisma.sql`
          SELECT
            COUNT(*)::int AS total,
            COUNT(DISTINCT user_id)::int AS contributors,
            COUNT(*) FILTER (WHERE user_id = ${userId})::int AS mine
          FROM care_events
          WHERE occurred_at >= NOW() - INTERVAL '7 days'
            AND pathway = ANY(${challenge.pathways}::text[])
        `);
        const aggregate = rows[0] ?? { total: 0, contributors: 0, mine: 0 };
        return {
          ...challenge,
          total: aggregate.total,
          contributors: aggregate.contributors,
          myContribution: aggregate.mine,
          progress: Math.min(1, aggregate.total / challenge.target),
          completed: aggregate.total >= challenge.target,
        };
      })
    );

    return {
      generatedAt: new Date().toISOString(),
      windowDays: 7,
      challenges,
      principles: [
        'everyone-contributes',
        'nobody-loses',
        'no-raw-distance-ranking',
        'no-medical-competition',
      ],
    };
  }
}
