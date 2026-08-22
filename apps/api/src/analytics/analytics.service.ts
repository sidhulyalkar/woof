import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

const DAY_MS = 24 * 60 * 60 * 1000;

type ScoreObservation = {
  score: number;
  label: number;
  source: 'baseline' | 'learned';
};

@Injectable()
export class AnalyticsService {
  constructor(private readonly prisma: PrismaService) {}

  async getNorthStarMetrics(timeframeMinutes: number) {
    const since = new Date(Date.now() - timeframeMinutes * 60 * 1000);
    const [funnel, calibration, userStats, dataYield] = await Promise.all([
      this.getRelationshipFunnel(since),
      this.getCompatibilityCalibration(since),
      this.getUserStats(since),
      this.calculateDataYieldPerUser(since),
    ]);

    return {
      objective: 'discovery_to_repeat_meetup',
      funnel,
      compatibilityCalibration: calibration.summary,
      totalUsers: userStats.total,
      activeUsers: userStats.active,
      dataYieldPerUser: dataYield,
      // Backwards-compatible fields for the existing portfolio dashboard.
      meetupConversionRate: funnel.conversationToCompletedMeetupRate,
      retention7Day: funnel.repeatMeetupRate,
      totalMeetups: funnel.meetupProposedPairs,
      completedMeetups: funnel.completedMeetupPairs,
    };
  }

  async getDetailedMetrics(timeframeMinutes: number) {
    const since = new Date(Date.now() - timeframeMinutes * 60 * 1000);
    const [funnel, calibration, eventFeedback, serviceIntents] = await Promise.all([
      this.getRelationshipFunnel(since),
      this.getCompatibilityCalibration(since),
      this.prisma.eventFeedback.aggregate({
        where: { createdAt: { gte: since } },
        _count: true,
        _avg: { vibeScore: true },
      }),
      this.prisma.serviceIntent.groupBy({
        by: ['action'],
        _count: true,
        where: { createdAt: { gte: since } },
      }),
    ]);

    return {
      funnel,
      calibration,
      eventsFeedback: {
        total: eventFeedback._count || 0,
        avgVibeScore: eventFeedback._avg?.vibeScore || 0,
      },
      serviceIntents: serviceIntents.map((entry) => ({
        action: entry.action,
        count: entry._count,
      })),
    };
  }

  async getRelationshipFunnel(since: Date) {
    const [scoreEvents, conversations, proposals] = await Promise.all([
      this.prisma.telemetry.findMany({
        where: {
          event: 'COMPATIBILITY_SCORED',
          createdAt: { gte: since },
        },
        select: { data: true, createdAt: true },
      }),
      this.prisma.conversation.findMany({
        where: { createdAt: { gte: since } },
        select: {
          id: true,
          createdAt: true,
          participants: { select: { userId: true } },
          messages: { select: { id: true }, take: 1 },
        },
      }),
      this.prisma.meetupProposal.findMany({
        where: { createdAt: { gte: since } },
        select: {
          proposerId: true,
          recipientId: true,
          status: true,
          occurredAt: true,
          createdAt: true,
        },
      }),
    ]);

    const petIds = new Set<string>();
    for (const event of scoreEvents) {
      const data = this.asRecord(event.data);
      if (typeof data.petAId === 'string') petIds.add(data.petAId);
      if (typeof data.petBId === 'string') petIds.add(data.petBId);
    }
    const pets = petIds.size
      ? await this.prisma.pet.findMany({
          where: { id: { in: [...petIds] } },
          select: { id: true, ownerId: true },
        })
      : [];
    const ownerByPet = new Map(pets.map((pet) => [pet.id, pet.ownerId]));

    const discoveryPairs = new Set<string>();
    for (const event of scoreEvents) {
      const data = this.asRecord(event.data);
      const ownerA = typeof data.petAId === 'string' ? ownerByPet.get(data.petAId) : undefined;
      const ownerB = typeof data.petBId === 'string' ? ownerByPet.get(data.petBId) : undefined;
      if (ownerA && ownerB && ownerA !== ownerB) discoveryPairs.add(this.pairKey(ownerA, ownerB));
    }

    const conversationPairs = new Set<string>();
    for (const conversation of conversations) {
      if (conversation.messages.length === 0 || conversation.participants.length !== 2) continue;
      conversationPairs.add(
        this.pairKey(conversation.participants[0].userId, conversation.participants[1].userId),
      );
    }

    const proposedPairs = new Set<string>();
    const completedPairCounts = new Map<string, number>();
    for (const proposal of proposals) {
      const pair = this.pairKey(proposal.proposerId, proposal.recipientId);
      proposedPairs.add(pair);
      if (proposal.occurredAt || proposal.status.toLowerCase() === 'completed') {
        completedPairCounts.set(pair, (completedPairCounts.get(pair) ?? 0) + 1);
      }
    }
    const completedPairs = new Set(completedPairCounts.keys());
    const repeatPairs = new Set(
      [...completedPairCounts.entries()].filter(([, count]) => count >= 2).map(([pair]) => pair),
    );

    const conversationsAfterDiscovery = this.intersection(conversationPairs, discoveryPairs);
    const proposalsAfterConversation = this.intersection(proposedPairs, conversationsAfterDiscovery);
    const completedAfterConversation = this.intersection(completedPairs, conversationsAfterDiscovery);
    const repeatAfterCompletion = this.intersection(repeatPairs, completedAfterConversation);

    return {
      discoveryPairs: discoveryPairs.size,
      conversationPairs: conversationsAfterDiscovery.size,
      meetupProposedPairs: proposalsAfterConversation.size,
      completedMeetupPairs: completedAfterConversation.size,
      repeatMeetupPairs: repeatAfterCompletion.size,
      discoveryToConversationRate: this.percent(
        conversationsAfterDiscovery.size,
        discoveryPairs.size,
      ),
      conversationToProposalRate: this.percent(
        proposalsAfterConversation.size,
        conversationsAfterDiscovery.size,
      ),
      conversationToCompletedMeetupRate: this.percent(
        completedAfterConversation.size,
        conversationsAfterDiscovery.size,
      ),
      completedToRepeatMeetupRate: this.percent(
        repeatAfterCompletion.size,
        completedAfterConversation.size,
      ),
      repeatMeetupRate: this.percent(repeatAfterCompletion.size, discoveryPairs.size),
    };
  }

  async getCompatibilityCalibration(since: Date) {
    const scoreEvents = await this.prisma.telemetry.findMany({
      where: { event: 'COMPATIBILITY_SCORED', createdAt: { gte: since } },
      select: { data: true, createdAt: true },
      orderBy: { createdAt: 'asc' },
      take: 5000,
    });

    const petIds = new Set<string>();
    for (const event of scoreEvents) {
      const data = this.asRecord(event.data);
      if (typeof data.petAId === 'string') petIds.add(data.petAId);
      if (typeof data.petBId === 'string') petIds.add(data.petBId);
    }
    const pets = petIds.size
      ? await this.prisma.pet.findMany({
          where: { id: { in: [...petIds] } },
          select: { id: true, ownerId: true },
        })
      : [];
    const ownerByPet = new Map(pets.map((pet) => [pet.id, pet.ownerId]));

    const outcomes = await this.prisma.meetupProposal.findMany({
      where: {
        createdAt: { gte: since },
        OR: [{ rating: { not: null } }, { occurredAt: { not: null } }],
      },
      select: {
        proposerId: true,
        recipientId: true,
        rating: true,
        occurredAt: true,
        createdAt: true,
      },
      orderBy: { createdAt: 'asc' },
    });
    const outcomesByPair = new Map<string, typeof outcomes>();
    for (const outcome of outcomes) {
      const key = this.pairKey(outcome.proposerId, outcome.recipientId);
      const existing = outcomesByPair.get(key) ?? [];
      existing.push(outcome);
      outcomesByPair.set(key, existing);
    }

    const observations: ScoreObservation[] = [];
    let fallbackCount = 0;
    let learnedAttemptCount = 0;
    let learnedLatencyTotal = 0;

    for (const event of scoreEvents) {
      const data = this.asRecord(event.data);
      if (data.fallbackReason) fallbackCount += 1;
      if (typeof data.learnedScore === 'number') {
        learnedAttemptCount += 1;
        if (typeof data.latencyMs === 'number') learnedLatencyTotal += data.latencyMs;
      }

      const ownerA = typeof data.petAId === 'string' ? ownerByPet.get(data.petAId) : undefined;
      const ownerB = typeof data.petBId === 'string' ? ownerByPet.get(data.petBId) : undefined;
      if (!ownerA || !ownerB || ownerA === ownerB) continue;
      const pairOutcomes = outcomesByPair.get(this.pairKey(ownerA, ownerB)) ?? [];
      const futureOutcome = pairOutcomes.find((outcome) => {
        const outcomeTime = outcome.occurredAt ?? outcome.createdAt;
        return (
          outcomeTime.getTime() > event.createdAt.getTime() &&
          outcomeTime.getTime() <= event.createdAt.getTime() + 30 * DAY_MS
        );
      });
      if (!futureOutcome || futureOutcome.rating === null) continue;
      const label = futureOutcome.rating >= 4 ? 1 : 0;

      if (typeof data.baselineScore === 'number') {
        observations.push({ score: data.baselineScore, label, source: 'baseline' });
      }
      if (typeof data.learnedScore === 'number') {
        observations.push({ score: data.learnedScore, label, source: 'learned' });
      }
    }

    const baseline = observations.filter((entry) => entry.source === 'baseline');
    const learned = observations.filter((entry) => entry.source === 'learned');
    const eventCount = scoreEvents.length;

    return {
      summary: {
        scoreEvents: eventCount,
        labeledFutureOutcomes: new Set(
          observations.map((entry, index) => `${entry.source}-${index}`),
        ).size,
        fallbackRate: this.percent(fallbackCount, eventCount),
        learnedAttemptRate: this.percent(learnedAttemptCount, eventCount),
        learnedMeanLatencyMs:
          learnedAttemptCount > 0 ? Math.round(learnedLatencyTotal / learnedAttemptCount) : 0,
      },
      baseline: this.calibrationSummary(baseline),
      learned: this.calibrationSummary(learned),
    };
  }

  async recordTelemetry(data: {
    userId?: string;
    source: string;
    event: string;
    metadata?: unknown;
  }) {
    const source = data.source.trim().slice(0, 64);
    const event = data.event.trim().toUpperCase().replace(/[^A-Z0-9_]/g, '_').slice(0, 80);
    return this.prisma.telemetry.create({
      data: {
        userId: data.userId,
        source,
        event,
        data: this.asRecord(data.metadata),
      },
    });
  }

  async getEventCounts(since: Date) {
    const events = await this.prisma.telemetry.groupBy({
      by: ['event'],
      _count: true,
      where: { createdAt: { gte: since } },
      orderBy: { _count: { event: 'desc' } },
    });
    return events.map((entry) => ({ event: entry.event, count: entry._count }));
  }

  async getUserActivity(userId: string, limit = 50) {
    return this.prisma.telemetry.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
      take: Math.max(1, Math.min(Number(limit) || 50, 100)),
    });
  }

  async getActiveUsersCount(since: Date) {
    const result = await this.prisma.telemetry.groupBy({
      by: ['userId'],
      where: { userId: { not: null }, event: 'APP_OPEN', createdAt: { gte: since } },
    });
    return result.length;
  }

  async getScreenViews(since: Date) {
    const rows = await this.prisma.telemetry.findMany({
      where: { event: 'SCREEN_VIEW', createdAt: { gte: since } },
      select: { data: true },
    });
    const counts = new Map<string, number>();
    for (const row of rows) {
      const screen = String(this.asRecord(row.data).screen ?? 'unknown');
      counts.set(screen, (counts.get(screen) ?? 0) + 1);
    }
    return [...counts.entries()].map(([screen, views]) => ({ screen, views }));
  }

  private calibrationSummary(observations: ScoreObservation[]) {
    if (observations.length === 0) {
      return { count: 0, brier: null, ece10: null, bins: [] };
    }

    const brier =
      observations.reduce((sum, entry) => sum + (entry.score - entry.label) ** 2, 0) /
      observations.length;
    const bins = Array.from({ length: 10 }, (_, index) => {
      const lower = index / 10;
      const upper = (index + 1) / 10;
      const values = observations.filter(
        (entry) => entry.score >= lower && (index === 9 ? entry.score <= upper : entry.score < upper),
      );
      if (values.length === 0) {
        return { lower, upper, count: 0, meanScore: null, positiveRate: null, gap: null };
      }
      const meanScore = values.reduce((sum, entry) => sum + entry.score, 0) / values.length;
      const positiveRate = values.reduce((sum, entry) => sum + entry.label, 0) / values.length;
      return {
        lower,
        upper,
        count: values.length,
        meanScore: this.round(meanScore),
        positiveRate: this.round(positiveRate),
        gap: this.round(Math.abs(meanScore - positiveRate)),
      };
    });
    const ece = bins.reduce(
      (sum, bin) => sum + (bin.count / observations.length) * (bin.gap ?? 0),
      0,
    );
    return {
      count: observations.length,
      brier: this.round(brier),
      ece10: this.round(ece),
      bins,
    };
  }

  private async calculateDataYieldPerUser(since: Date) {
    const activeUsers = await this.prisma.telemetry.groupBy({
      by: ['userId'],
      where: { userId: { not: null }, createdAt: { gte: since } },
    });
    if (activeUsers.length === 0) return 0;
    const [meetupFeedback, eventFeedback] = await Promise.all([
      this.prisma.meetupProposal.count({
        where: { rating: { not: null }, createdAt: { gte: since } },
      }),
      this.prisma.eventFeedback.count({ where: { createdAt: { gte: since } } }),
    ]);
    return this.round((meetupFeedback + eventFeedback) / activeUsers.length);
  }

  private async getUserStats(since: Date) {
    const [total, activeRows] = await Promise.all([
      this.prisma.user.count(),
      this.prisma.telemetry.groupBy({
        by: ['userId'],
        where: { userId: { not: null }, createdAt: { gte: since } },
      }),
    ]);
    return { total, active: activeRows.length };
  }

  private asRecord(value: unknown): Record<string, unknown> {
    return value && typeof value === 'object' && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : {};
  }

  private pairKey(a: string, b: string) {
    return [a, b].sort().join('::');
  }

  private intersection(a: Set<string>, b: Set<string>) {
    return new Set([...a].filter((value) => b.has(value)));
  }

  private percent(numerator: number, denominator: number) {
    if (denominator === 0) return 0;
    return this.round((numerator / denominator) * 100);
  }

  private round(value: number) {
    return Math.round(value * 1000) / 1000;
  }
}
