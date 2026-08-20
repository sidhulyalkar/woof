import { Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { RecommendationFeedbackDto } from './dto/recommendation-feedback.dto';

const ALGORITHM_VERSION = 'next-best-action-v1';
const RELATIONSHIP_VERSION = 'relationship-signals-v1';
const DAY_MS = 24 * 60 * 60 * 1000;

type RecommendationCategory =
  | 'activity'
  | 'enrichment'
  | 'social'
  | 'recovery'
  | 'reflection'
  | 'goal';

type Recommendation = {
  id: string;
  category: RecommendationCategory;
  title: string;
  reason: string;
  actionLabel: string;
  href: string;
  score: number;
  confidence: number;
  evidence: string[];
};

type TemperamentVector = {
  energy: number | null;
  sociability: number | null;
  caution: number | null;
  trainability: number | null;
  coverage: number;
};

@Injectable()
export class InsightsService {
  constructor(private readonly prisma: PrismaService) {}

  async getForUser(userId: string, requestedPetId?: string) {
    const pet = requestedPetId
      ? await this.prisma.pet.findFirst({
          where: { id: requestedPetId, ownerId: userId },
          select: {
            id: true,
            name: true,
            species: true,
            breed: true,
            birthdate: true,
            temperament: true,
            avatarUrl: true,
          },
        })
      : await this.prisma.pet.findFirst({
          where: { ownerId: userId },
          orderBy: { createdAt: 'asc' },
          select: {
            id: true,
            name: true,
            species: true,
            breed: true,
            birthdate: true,
            temperament: true,
            avatarUrl: true,
          },
        });

    if (!pet) {
      throw new NotFoundException('No pet profile is available for these insights');
    }

    const now = new Date();
    const twentyEightDaysAgo = new Date(now.getTime() - 28 * DAY_MS);
    const ninetyDaysAgo = new Date(now.getTime() - 90 * DAY_MS);

    const [activities, goals, quiz, meetups, feedbackHistory] = await Promise.all([
      this.prisma.activity.findMany({
        where: {
          userId,
          petId: pet.id,
          startedAt: { gte: twentyEightDaysAgo },
        },
        orderBy: { startedAt: 'desc' },
        select: {
          id: true,
          type: true,
          startedAt: true,
          endedAt: true,
        },
      }),
      this.prisma.mutualGoal.findMany({
        where: { userId, petId: pet.id, status: 'ACTIVE' },
        orderBy: { endDate: 'asc' },
        take: 3,
        select: {
          id: true,
          goalType: true,
          targetNumber: true,
          targetUnit: true,
          currentValue: true,
          progress: true,
          endDate: true,
        },
      }),
      this.prisma.quizResponse.findFirst({
        where: { userId, OR: [{ petId: pet.id }, { petId: null }] },
        orderBy: { completedAt: 'desc' },
        select: { completedAt: true },
      }),
      this.prisma.meetupProposal.findMany({
        where: {
          OR: [{ proposerId: userId }, { recipientId: userId }],
          createdAt: { gte: ninetyDaysAgo },
        },
        orderBy: { createdAt: 'desc' },
        take: 30,
        select: {
          status: true,
          occurredAt: true,
          rating: true,
          feedbackTags: true,
          createdAt: true,
        },
      }),
      this.prisma.telemetry.findMany({
        where: {
          userId,
          petId: pet.id,
          source: 'INSIGHTS',
          createdAt: { gte: ninetyDaysAgo },
        },
        orderBy: { createdAt: 'desc' },
        take: 200,
        select: { event: true, data: true },
      }),
    ]);

    const temperament = this.buildTemperamentVector(pet.temperament);
    const activitySummary = this.summarizeActivity(activities, now);
    const adaptiveWeights = this.buildAdaptiveWeights(feedbackHistory);
    const profileConfidence = this.calculateProfileConfidence({
      hasBirthdate: Boolean(pet.birthdate),
      temperamentCoverage: temperament.coverage,
      activityCount: activities.length,
      hasQuiz: Boolean(quiz),
      meetupFeedbackCount: meetups.filter((meetup) => meetup.rating !== null).length,
    });

    const recommendations = this.rankRecommendations({
      petName: pet.name,
      activitySummary,
      temperament,
      goals,
      quizCompletedAt: quiz?.completedAt ?? null,
      meetups,
      adaptiveWeights,
      profileConfidence,
    });

    const relationshipSignals = this.buildRelationshipSignals({
      activities,
      activitySummary,
      meetups,
      hasQuiz: Boolean(quiz),
    });

    return {
      pet: {
        id: pet.id,
        name: pet.name,
        species: pet.species,
        avatarUrl: pet.avatarUrl,
      },
      generatedAt: now.toISOString(),
      algorithm: {
        recommendations: ALGORITHM_VERSION,
        relationshipSignals: RELATIONSHIP_VERSION,
        confidence: profileConfidence,
        principles: [
          'individual-behavior-first',
          'recent-context-aware',
          'owner-outcome-adaptive',
          'non-medical-wellness-guidance',
        ],
      },
      recommendations,
      relationshipSignals,
      learningSummary: this.buildLearningSummary({
        petName: pet.name,
        activities,
        activitySummary,
        temperament,
        meetups,
      }),
      disclaimer:
        'Woof uses observed routines and owner-provided context for general wellness and relationship guidance. It does not diagnose conditions or replace veterinary advice.',
    };
  }

  async recordFeedback(
    userId: string,
    petId: string,
    dto: RecommendationFeedbackDto,
  ) {
    const pet = await this.prisma.pet.findFirst({
      where: { id: petId, ownerId: userId },
      select: { id: true },
    });

    if (!pet) {
      throw new NotFoundException('Pet not found');
    }

    return this.prisma.telemetry.create({
      data: {
        source: 'INSIGHTS',
        event: `RECOMMENDATION_${dto.outcome.toUpperCase()}`,
        userId,
        petId,
        data: {
          recommendationId: dto.recommendationId,
          category: dto.category ?? null,
          algorithmVersion: ALGORITHM_VERSION,
        } as Prisma.InputJsonValue,
      },
      select: { id: true, event: true, createdAt: true },
    });
  }

  private rankRecommendations(input: {
    petName: string;
    activitySummary: ReturnType<InsightsService['summarizeActivity']>;
    temperament: TemperamentVector;
    goals: Array<{
      id: string;
      goalType: string;
      targetNumber: number;
      targetUnit: string;
      currentValue: number;
      progress: number;
      endDate: Date;
    }>;
    quizCompletedAt: Date | null;
    meetups: Array<{
      status: string;
      occurredAt: Date | null;
      rating: number | null;
      feedbackTags: string[];
      createdAt: Date;
    }>;
    adaptiveWeights: Record<string, number>;
    profileConfidence: number;
  }): Recommendation[] {
    const {
      petName,
      activitySummary,
      temperament,
      goals,
      quizCompletedAt,
      meetups,
      adaptiveWeights,
      profileConfidence,
    } = input;
    const candidates: Recommendation[] = [];

    if (goals.length > 0) {
      const goal = goals[0];
      const progress = Math.max(0, Math.min(1, goal.progress || 0));
      candidates.push({
        id: `goal-${goal.id}`,
        category: 'goal',
        title: `Keep your shared ${goal.goalType.toLowerCase()} goal moving`,
        reason: `${petName}'s active goal is ${Math.round(progress * 100)}% complete. A small shared session can preserve continuity without chasing a rigid streak.`,
        actionLabel: 'View shared goal',
        href: '/leaderboard',
        score: 0.76 + 0.12 * (1 - progress),
        confidence: profileConfidence,
        evidence: ['active mutual goal', 'recent goal progress'],
      });
    }

    const movementNeed = Math.max(
      activitySummary.daysSinceLatest > 2 ? 0.78 : 0.48,
      0.72 - activitySummary.activeDays28 / 35,
    );
    candidates.push({
      id: 'activity-shared-movement',
      category: 'activity',
      title: `Make movement time together`,
      reason:
        activitySummary.activeDays28 === 0
          ? `Woof has not seen a recent shared activity with ${petName} yet. Logging the next walk, play session, run, or hike gives the learning system a useful baseline.`
          : `${petName} has ${activitySummary.activeDays28} active days recorded in the last four weeks. The next shared activity helps Woof learn the rhythm that works for both of you.`,
      actionLabel: 'Start or log activity',
      href: '/activity',
      score: movementNeed,
      confidence: Math.max(0.42, profileConfidence),
      evidence: ['28-day activity history', 'activity recency'],
    });

    const energy = temperament.energy ?? 0.5;
    const enrichmentNeed =
      0.58 + 0.16 * energy + (activitySummary.uniqueTypes <= 1 ? 0.14 : 0);
    candidates.push({
      id: 'enrichment-variety',
      category: 'enrichment',
      title: `Add a little novelty`,
      reason: `${petName}'s recent routine includes ${activitySummary.uniqueTypes || 'no'} distinct activity type${activitySummary.uniqueTypes === 1 ? '' : 's'}. A short training, sniffing, puzzle, or exploratory session can add cognitive variety without assuming every dog needs the same routine.`,
      actionLabel: 'Explore an activity',
      href: '/activity',
      score: enrichmentNeed,
      confidence: Math.max(0.4, (profileConfidence + temperament.coverage) / 2),
      evidence: ['activity variety', 'owner-reported temperament'],
    });

    const sociability = temperament.sociability;
    const caution = temperament.caution;
    if (sociability !== null && (caution === null || caution < 0.72)) {
      const lowPressure = caution !== null && caution > 0.38;
      const positiveMeetups = meetups.filter(
        (meetup) =>
          meetup.rating !== null &&
          meetup.rating >= 4 &&
          !meetup.feedbackTags.includes('temperament'),
      ).length;
      candidates.push({
        id: 'social-compatible-context',
        category: 'social',
        title: lowPressure ? 'Try a low-pressure social setting' : 'Find a compatible social moment',
        reason: lowPressure
          ? `${petName}'s profile suggests some social interest with a reason to keep introductions gradual. Prefer space, easy exits, and owner-controlled pacing.`
          : `${petName}'s profile contains positive social signals${positiveMeetups > 0 ? ` and ${positiveMeetups} recent positive meetup outcome${positiveMeetups === 1 ? '' : 's'}` : ''}. Compatibility-first discovery can help you choose a better context than a random encounter.`,
        actionLabel: 'See compatible pets',
        href: '/discover',
        score: 0.48 + 0.24 * sociability + Math.min(0.12, positiveMeetups * 0.04),
        confidence: Math.max(0.38, (profileConfidence + temperament.coverage) / 2),
        evidence: ['individual social behavior profile', 'meetup outcomes when available'],
      });
    }

    if (activitySummary.recentLoadRatio > 1.5 && activitySummary.recentCount >= 3) {
      candidates.push({
        id: 'recovery-balance',
        category: 'recovery',
        title: 'Balance a busier stretch',
        reason: `${petName}'s recorded activity has been busier than the preceding weeks. A quieter enrichment or decompression choice can add balance. If the change reflects pain, fatigue, or another health concern, use a veterinarian rather than an app recommendation.`,
        actionLabel: 'Review recent activity',
        href: '/activity',
        score: 0.74,
        confidence: Math.min(0.92, profileConfidence + 0.08),
        evidence: ['7-day activity load', 'prior 21-day baseline'],
      });
    }

    const completedMeetupsWithoutFeedback = meetups.filter(
      (meetup) =>
        (meetup.status.toLowerCase() === 'completed' || meetup.occurredAt) &&
        meetup.rating === null,
    ).length;
    const quizStale =
      !quizCompletedAt || Date.now() - quizCompletedAt.getTime() > 60 * DAY_MS;
    if (completedMeetupsWithoutFeedback > 0 || quizStale) {
      candidates.push({
        id: 'reflection-update-profile',
        category: 'reflection',
        title: `Teach Woof what you noticed`,
        reason:
          completedMeetupsWithoutFeedback > 0
            ? `There ${completedMeetupsWithoutFeedback === 1 ? 'is' : 'are'} ${completedMeetupsWithoutFeedback} recent shared social experience${completedMeetupsWithoutFeedback === 1 ? '' : 's'} without feedback. A quick reflection is more valuable to personalization than another passive click.`
            : `A quick preference refresh helps Woof distinguish stable traits from a routine that has changed over time.`,
        actionLabel: 'Review preferences',
        href: '/profile',
        score: 0.7,
        confidence: 0.82,
        evidence: ['owner reflection recency', 'completed meetup feedback'],
      });
    }

    return candidates
      .map((candidate) => ({
        ...candidate,
        score: this.clamp01(
          candidate.score * (adaptiveWeights[candidate.category] ?? 1),
        ),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 4);
  }

  private summarizeActivity(
    activities: Array<{
      type: string;
      startedAt: Date;
      endedAt: Date | null;
    }>,
    now: Date,
  ) {
    const dayKeys = new Set(
      activities.map((activity) => activity.startedAt.toISOString().slice(0, 10)),
    );
    const uniqueTypes = new Set(
      activities.map((activity) => activity.type.toUpperCase()),
    ).size;
    const latest = activities[0]?.startedAt ?? null;
    const recentBoundary = new Date(now.getTime() - 7 * DAY_MS);
    const priorBoundary = new Date(now.getTime() - 28 * DAY_MS);
    const recent = activities.filter((activity) => activity.startedAt >= recentBoundary);
    const prior = activities.filter(
      (activity) =>
        activity.startedAt >= priorBoundary && activity.startedAt < recentBoundary,
    );
    const durationMinutes = (items: typeof activities) =>
      items.reduce((sum, activity) => {
        if (!activity.endedAt) return sum;
        return (
          sum +
          Math.max(0, activity.endedAt.getTime() - activity.startedAt.getTime()) /
            60000
        );
      }, 0);
    const recentMinutes = durationMinutes(recent);
    const priorWeeklyMinutes = durationMinutes(prior) / 3;

    return {
      activeDays28: dayKeys.size,
      uniqueTypes,
      daysSinceLatest: latest
        ? Math.floor((now.getTime() - latest.getTime()) / DAY_MS)
        : 99,
      recentCount: recent.length,
      recentMinutes,
      priorWeeklyMinutes,
      recentLoadRatio:
        priorWeeklyMinutes > 10
          ? recentMinutes / priorWeeklyMinutes
          : recent.length >= 3
            ? 1
            : 0,
    };
  }

  private buildRelationshipSignals(input: {
    activities: Array<{ type: string; startedAt: Date; endedAt: Date | null }>;
    activitySummary: ReturnType<InsightsService['summarizeActivity']>;
    meetups: Array<{
      status: string;
      occurredAt: Date | null;
      rating: number | null;
      feedbackTags: string[];
      createdAt: Date;
    }>;
    hasQuiz: boolean;
  }) {
    const { activities, activitySummary, meetups, hasQuiz } = input;
    const completedMeetups = meetups.filter(
      (meetup) => meetup.status.toLowerCase() === 'completed' || meetup.occurredAt,
    );
    const reflectedMeetups = completedMeetups.filter(
      (meetup) => meetup.rating !== null || meetup.feedbackTags.length > 0,
    );

    return [
      {
        key: 'consistency',
        label: 'Shared routine',
        value: Math.round(this.clamp01(activitySummary.activeDays28 / 14) * 100),
        explanation: `${activitySummary.activeDays28} days with a recorded shared activity in the last four weeks.`,
      },
      {
        key: 'variety',
        label: 'Experience variety',
        value: Math.round(this.clamp01(activitySummary.uniqueTypes / 4) * 100),
        explanation: `${activitySummary.uniqueTypes} distinct movement or play categories are represented in recent activity.`,
      },
      {
        key: 'shared-experience',
        label: 'Shared experiences',
        value: Math.round(
          this.clamp01((activities.length + completedMeetups.length * 2) / 16) * 100,
        ),
        explanation: `${activities.length} recent activities and ${completedMeetups.length} completed social experiences contribute context.`,
      },
      {
        key: 'reflection',
        label: 'Learning feedback',
        value: Math.round(
          this.clamp01(((hasQuiz ? 1 : 0) + reflectedMeetups.length) / 4) * 100,
        ),
        explanation:
          'Preference updates and post-experience feedback help Woof learn what works rather than infer intent from engagement alone.',
      },
    ];
  }

  private buildLearningSummary(input: {
    petName: string;
    activities: Array<{ type: string }>;
    activitySummary: ReturnType<InsightsService['summarizeActivity']>;
    temperament: TemperamentVector;
    meetups: Array<{ rating: number | null; feedbackTags: string[] }>;
  }) {
    const { petName, activities, activitySummary, temperament, meetups } = input;
    const favoriteType = this.mode(activities.map((activity) => activity.type));
    const positiveMeetups = meetups.filter(
      (meetup) => meetup.rating !== null && meetup.rating >= 4,
    ).length;
    const observations: string[] = [];

    if (favoriteType) {
      observations.push(
        `${favoriteType.toLowerCase()} is the most frequently recorded recent activity with ${petName}.`,
      );
    }
    if (activitySummary.activeDays28 > 0) {
      observations.push(
        `Woof has ${activitySummary.activeDays28} active-day observations from the last four weeks.`,
      );
    }
    if (temperament.sociability !== null) {
      observations.push(
        `The current behavior profile provides ${Math.round(temperament.coverage * 100)}% coverage for beta personalization dimensions.`,
      );
    }
    if (positiveMeetups > 0) {
      observations.push(
        `${positiveMeetups} recent meetup outcome${positiveMeetups === 1 ? '' : 's'} received a positive rating.`,
      );
    }

    return observations.slice(0, 3);
  }

  private buildTemperamentVector(value: Prisma.JsonValue | null): TemperamentVector {
    const traits = new Map<string, number>();

    if (Array.isArray(value)) {
      for (const item of value) {
        if (typeof item === 'string') {
          traits.set(this.normalizeKey(item), 1);
        }
      }
    } else if (value && typeof value === 'object') {
      for (const [key, raw] of Object.entries(value)) {
        if (typeof raw === 'number') {
          traits.set(this.normalizeKey(key), this.clamp01((raw - 1) / 4));
        } else if (typeof raw === 'boolean') {
          traits.set(this.normalizeKey(key), raw ? 1 : 0);
        }
      }
    }

    const trait = (...aliases: string[]) => {
      const values = aliases
        .map((alias) => traits.get(this.normalizeKey(alias)))
        .filter((entry): entry is number => entry !== undefined);
      if (values.length === 0) return null;
      return values.reduce((sum, entry) => sum + entry, 0) / values.length;
    };

    const energyPositive = trait('energetic', 'energy', 'playful', 'active');
    const calm = trait('calm');
    const socialPositive = trait('friendly', 'social', 'sociable');
    const shy = trait('shy');
    const cautionPositive = trait(
      'shy',
      'fearful',
      'anxious',
      'dog directed fear',
      'protective',
    );
    const trainable = trait('trainable', 'trainability');

    const combine = (...values: Array<number | null>) => {
      const present = values.filter((entry): entry is number => entry !== null);
      if (present.length === 0) return null;
      return present.reduce((sum, entry) => sum + entry, 0) / present.length;
    };

    const dimensions = [
      energyPositive !== null || calm !== null
        ? combine(energyPositive, calm === null ? null : 1 - calm)
        : null,
      socialPositive !== null || shy !== null
        ? combine(socialPositive, shy === null ? null : 1 - shy)
        : null,
      cautionPositive,
      trainable !== null || calm !== null ? combine(trainable, calm) : null,
    ];

    return {
      energy: dimensions[0],
      sociability: dimensions[1],
      caution: dimensions[2],
      trainability: dimensions[3],
      coverage: dimensions.filter((entry) => entry !== null).length / dimensions.length,
    };
  }

  private buildAdaptiveWeights(
    feedback: Array<{ event: string; data: Prisma.JsonValue | null }>,
  ) {
    const stats: Record<string, { positive: number; total: number }> = {};
    for (const item of feedback) {
      const category = this.jsonString(item.data, 'category');
      if (!category) continue;
      stats[category] ??= { positive: 0, total: 0 };
      if (item.event === 'RECOMMENDATION_SHOWN') continue;
      stats[category].total += 1;
      if (
        item.event === 'RECOMMENDATION_ACCEPTED' ||
        item.event === 'RECOMMENDATION_COMPLETED'
      ) {
        stats[category].positive += 1;
      }
    }

    return Object.fromEntries(
      Object.entries(stats).map(([category, stat]) => {
        const posterior = (stat.positive + 1.5) / (stat.total + 3);
        return [category, 0.8 + posterior * 0.4];
      }),
    );
  }

  private calculateProfileConfidence(input: {
    hasBirthdate: boolean;
    temperamentCoverage: number;
    activityCount: number;
    hasQuiz: boolean;
    meetupFeedbackCount: number;
  }) {
    const activityCoverage = this.clamp01(input.activityCount / 12);
    const meetupCoverage = this.clamp01(input.meetupFeedbackCount / 4);
    return this.clamp01(
      0.18 +
        (input.hasBirthdate ? 0.12 : 0) +
        input.temperamentCoverage * 0.3 +
        activityCoverage * 0.22 +
        (input.hasQuiz ? 0.1 : 0) +
        meetupCoverage * 0.08,
    );
  }

  private jsonString(value: Prisma.JsonValue | null, key: string) {
    if (!value || Array.isArray(value) || typeof value !== 'object') return null;
    const raw = value[key];
    return typeof raw === 'string' ? raw : null;
  }

  private normalizeKey(value: string) {
    return value.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  }

  private clamp01(value: number) {
    return Math.max(0, Math.min(1, value));
  }

  private mode(values: string[]) {
    if (values.length === 0) return null;
    const counts = new Map<string, number>();
    let winner = values[0];
    for (const value of values) {
      const count = (counts.get(value) ?? 0) + 1;
      counts.set(value, count);
      if (count > (counts.get(winner) ?? 0)) winner = value;
    }
    return winner;
  }
}
