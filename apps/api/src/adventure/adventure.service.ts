import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { createHash } from 'node:crypto';
import { CareEventsService } from '../care-events/care-events.service';
import {
  QUEST_EVENT_TYPES,
  WELLBEING_PATHWAYS,
  type AdventureLearningCareEvent,
  type CareSummary,
  type WellbeingPathway,
} from '../care-events/care-event.types';
import { baseXpForEvent } from '../care-events/reward-policy';
import { InsightsService } from '../insights/insights.service';
import {
  ADVENTURE_LEARNING_POLICY_VERSION,
  deriveAdventureLearningSignals,
  type AdventureLearningSignals,
} from './adventure-learning-policy';
import { PrismaService } from '../prisma/prisma.service';
import { CompleteQuestDto } from './dto/adventure.dto';

type Quest = {
  id: string;
  key: string;
  title: string;
  description: string;
  why: string;
  primaryPathway: WellbeingPathway;
  pathways: WellbeingPathway[];
  xp: number;
  confidence: number;
  href: string;
  actionLabel: string;
  variant: 'recommended' | 'alternative' | 'wildcard';
  safeStopEligible: boolean;
  personalRelevance: number;
  expiresAt: string;
};

type Candidate = Omit<Quest, 'id' | 'variant' | 'expiresAt'> & { score: number };

type QuestCompletionContext = Pick<
  Quest,
  'id' | 'key' | 'title' | 'primaryPathway' | 'safeStopEligible' | 'personalRelevance'
>;

type SelectedQuestSnapshot = QuestCompletionContext;

const DAY_MS = 24 * 60 * 60 * 1000;

const CATEGORY_PATHWAYS: Record<string, WellbeingPathway> = {
  activity: 'MOVE',
  enrichment: 'ENRICH',
  social: 'CONNECT',
  recovery: 'RECOVER',
  reflection: 'BOND',
  goal: 'BOND',
};

@Injectable()
export class AdventureService {
  private readonly logger = new Logger(AdventureService.name);

  constructor(
    private readonly prisma: PrismaService,
    private readonly insights: InsightsService,
    private readonly careEvents: CareEventsService
  ) {}

  async getDashboard(userId: string, requestedPetId?: string) {
    const insights = await this.insights.getForUser(userId, requestedPetId);
    const [summary, learningEvents] = await Promise.all([
      this.careEvents.getSummary(userId, insights.pet.id),
      this.careEvents.getAdventureLearningEvents(userId, insights.pet.id),
    ]);
    const quests = this.buildQuests(userId, insights, summary, learningEvents);

    return {
      pet: insights.pet,
      generatedAt: new Date().toISOString(),
      bondXp: summary.bondXp,
      rhythm: summary.rhythm,
      compass: summary.pathways,
      quests,
      learningSummary: insights.learningSummary,
      principles: [
        'individual-fit-over-volume',
        'choice-over-coercion',
        'recovery-counts',
        'safe-opt-outs-succeed',
        'reward-based-learning',
      ],
      disclaimer:
        'The Pawprint Compass shows recent opportunity coverage, not a health score or veterinary assessment.',
    };
  }

  async recordInteraction(
    userId: string,
    petId: string,
    questId: string,
    interaction: 'SELECTED' | 'DISMISSED'
  ) {
    const dashboard = await this.getDashboard(userId, petId);
    const quest = dashboard.quests.find((candidate) => candidate.id === questId);
    if (!quest) throw new NotFoundException('This quest is no longer available');

    await this.careEvents.recordQuestInteraction({
      userId,
      petId,
      questId,
      interaction,
      pathway: quest.primaryPathway,
      context: {
        questKey: quest.key,
        confidence: quest.confidence,
        learningPolicyVersion: ADVENTURE_LEARNING_POLICY_VERSION,
        ...(interaction === 'SELECTED' ? { questSnapshot: this.questSnapshot(quest) } : {}),
      },
    });
    return { ok: true };
  }

  async completeQuest(userId: string, questId: string, dto: CompleteQuestDto) {
    const dashboard = await this.getDashboard(userId, dto.petId);
    const currentQuest = dashboard.quests.find((candidate) => candidate.id === questId);
    const selected = currentQuest
      ? null
      : await this.careEvents.getRecentSelectedQuestContext(userId, dto.petId, questId);
    const quest: QuestCompletionContext | null =
      currentQuest ?? this.questFromSnapshot(selected?.context?.questSnapshot, questId);

    if (!quest) throw new NotFoundException('This quest is no longer available');

    const safeOptOut = Boolean(dto.safeOptOut && quest.safeStopEligible);
    const learnedMismatch = dto.dogExperience === 'not_their_thing' && !safeOptOut;
    const eventType = safeOptOut
      ? 'SAFE_OPT_OUT'
      : learnedMismatch
        ? QUEST_EVENT_TYPES.BOND
        : QUEST_EVENT_TYPES[quest.primaryPathway];
    const rewardPathway: WellbeingPathway =
      safeOptOut || learnedMismatch ? 'BOND' : quest.primaryPathway;

    // Memory bonuses require a real, completed private asset owned by this exact
    // dog-owner pair. A random client-supplied asset ID never changes rewards.
    const verifiedMemory = dto.memoryAssetId
      ? await this.prisma.mediaAsset.findFirst({
          where: {
            id: dto.memoryAssetId,
            ownerId: userId,
            petId: dto.petId,
            status: 'READY',
          },
          select: { id: true },
        })
      : null;

    const receipt = await this.careEvents.record({
      userId,
      petId: dto.petId,
      eventType,
      pathway: rewardPathway,
      source: 'QUEST_ENGINE',
      evidenceType: 'SELF_REPORT',
      evidenceConfidence: 0.68,
      dedupeKey: `quest:${quest.id}`,
      safetyEligible: true,
      context: {
        questId: quest.id,
        questKey: quest.key,
        questTitle: quest.title,
        originalPathway: quest.primaryPathway,
        learningPolicyVersion: ADVENTURE_LEARNING_POLICY_VERSION,
        personalRelevance: quest.personalRelevance,
        memoryAdded: Boolean(verifiedMemory),
        memoryAssetId: verifiedMemory?.id ?? null,
      },
      outcome: {
        dogExperience: dto.dogExperience,
        ownerExperience: dto.ownerExperience,
        safeOptOut,
        note: dto.note ?? null,
      },
    });

    // These records are useful for analytics and continuity, but the CareEvent +
    // RewardLedger transaction above is authoritative. A telemetry outage must never
    // turn a successful reward into an apparent client failure. The upsert also lets
    // a retry repair a previously failed interaction write without issuing XP twice.
    try {
      await this.careEvents.recordQuestInteraction({
        userId,
        petId: dto.petId,
        questId,
        interaction: 'COMPLETED',
        pathway: rewardPathway,
        context: {
          questKey: quest.key,
          originalPathway: quest.primaryPathway,
          rewardPathway,
          learningPolicyVersion: ADVENTURE_LEARNING_POLICY_VERSION,
          dogExperience: dto.dogExperience,
          ownerExperience: dto.ownerExperience,
          safeOptOut,
          bondXp: receipt.bondXp,
        },
      });
    } catch (error) {
      this.logger.warn(
        `Quest ${questId} reward committed but interaction telemetry failed: ${
          error instanceof Error ? error.message : 'unknown error'
        }`
      );
    }

    try {
      await this.prisma.telemetry.create({
        data: {
          source: 'ADVENTURE',
          event: safeOptOut
            ? 'QUEST_SAFE_OPT_OUT'
            : learnedMismatch
              ? 'QUEST_LEARNED_MISMATCH'
              : 'QUEST_COMPLETED',
          userId,
          petId: dto.petId,
          data: {
            questId,
            pathway: rewardPathway,
            originalPathway: quest.primaryPathway,
            rewardPathway,
            learningPolicyVersion: ADVENTURE_LEARNING_POLICY_VERSION,
            bondXp: receipt.bondXp,
            duplicate: receipt.duplicate,
            dogExperience: dto.dogExperience,
            ownerExperience: dto.ownerExperience,
          },
        },
      });
    } catch (error) {
      this.logger.warn(
        `Quest ${questId} reward committed but Adventure telemetry failed: ${
          error instanceof Error ? error.message : 'unknown error'
        }`
      );
    }

    return {
      reward: receipt,
      message: safeOptOut
        ? 'You listened. Giving space was the right play.'
        : learnedMismatch
          ? 'Useful discovery. Woof will treat this as preference evidence, not a failure.'
          : dto.dogExperience === 'loved_it'
            ? 'That one is worth remembering.'
            : 'Nice read. Another useful piece of your shared pattern.',
    };
  }

  private questSnapshot(quest: Quest): SelectedQuestSnapshot {
    return {
      id: quest.id,
      key: quest.key,
      title: quest.title,
      primaryPathway: quest.primaryPathway,
      safeStopEligible: quest.safeStopEligible,
      personalRelevance: quest.personalRelevance,
    };
  }

  private questFromSnapshot(value: unknown, questId: string): SelectedQuestSnapshot | null {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;

    const candidate = value as Record<string, unknown>;
    const personalRelevance = candidate.personalRelevance;
    if (
      candidate.id !== questId ||
      typeof candidate.key !== 'string' ||
      candidate.key.length === 0 ||
      typeof candidate.title !== 'string' ||
      candidate.title.length === 0 ||
      !this.isWellbeingPathway(candidate.primaryPathway) ||
      typeof candidate.safeStopEligible !== 'boolean' ||
      typeof personalRelevance !== 'number' ||
      !Number.isFinite(personalRelevance) ||
      personalRelevance < 0.9 ||
      personalRelevance > 1.08
    ) {
      return null;
    }

    return {
      id: questId,
      key: candidate.key,
      title: candidate.title,
      primaryPathway: candidate.primaryPathway,
      safeStopEligible: candidate.safeStopEligible,
      personalRelevance,
    };
  }

  private isWellbeingPathway(value: unknown): value is WellbeingPathway {
    return (
      typeof value === 'string' &&
      WELLBEING_PATHWAYS.includes(value as (typeof WELLBEING_PATHWAYS)[number])
    );
  }

  private buildQuests(
    userId: string,
    insights: Awaited<ReturnType<InsightsService['getForUser']>>,
    summary: CareSummary,
    learningEvents: AdventureLearningCareEvent[]
  ): Quest[] {
    const dateKey = new Date().toISOString().slice(0, 10);
    const coverage = new Map(summary.pathways.map((item) => [item.pathway, item.coverage]));
    const learning = deriveAdventureLearningSignals(learningEvents);

    const candidates: Candidate[] = insights.recommendations.map((recommendation) => {
      const pathway = CATEGORY_PATHWAYS[recommendation.category] ?? 'BOND';
      const eventType = QUEST_EVENT_TYPES[pathway];
      const gap = 1 - (coverage.get(pathway) ?? 0) / 100;
      const personalRelevance = this.pathwayRelevance(learning, pathway);
      return {
        key: `insight-${recommendation.id}`,
        title: recommendation.title,
        description: this.questDescription(pathway, insights.pet.name),
        why: recommendation.reason,
        primaryPathway: pathway,
        pathways: this.secondaryPathways(pathway),
        xp: baseXpForEvent(eventType),
        confidence: recommendation.confidence,
        href: recommendation.href,
        actionLabel: recommendation.actionLabel,
        safeStopEligible: pathway === 'CONNECT' || pathway === 'LEARN',
        personalRelevance,
        score: recommendation.score * 0.68 + gap * 0.24 + (personalRelevance - 0.9) * 0.5,
      };
    });

    const templates: Candidate[] = [
      {
        key: 'sniffari',
        title: 'Take a Sniffari',
        description: `Give ${insights.pet.name} an easy exploration where sniffing, pausing, and choosing the pace are part of the point.`,
        why: 'Novel sensory exploration can add variety without turning distance into the objective.',
        primaryPathway: 'EXPLORE',
        pathways: ['EXPLORE', 'ENRICH', 'BOND'],
        xp: baseXpForEvent('QUEST_EXPLORE'),
        confidence: Math.max(0.55, insights.algorithm.confidence),
        href: '/activity',
        actionLabel: 'Start an exploration',
        safeStopEligible: false,
        personalRelevance: this.pathwayRelevance(learning, 'EXPLORE'),
        score: 0.62 + (1 - (coverage.get('EXPLORE') ?? 0) / 100) * 0.28,
      },
      {
        key: 'skill-spark',
        title: 'Five-minute skill spark',
        description:
          'Choose one tiny reward-based skill and stop while the session still feels easy.',
        why: 'Short comfortable repetitions build communication without making training a grind.',
        primaryPathway: 'LEARN',
        pathways: ['LEARN', 'BOND'],
        xp: baseXpForEvent('QUEST_LEARN'),
        confidence: 0.82,
        href: '/coach',
        actionLabel: 'Open Coach',
        safeStopEligible: true,
        personalRelevance: this.pathwayRelevance(learning, 'LEARN'),
        score: 0.58 + (1 - (coverage.get('LEARN') ?? 0) / 100) * 0.3,
      },
      {
        key: 'easy-day',
        title: 'Choose an easy day',
        description:
          'Keep the outing simple, add calm sniffing or enrichment, and leave room for decompression.',
        why: 'Recovery is a valid wellbeing action. Woof does not treat more miles as automatically better.',
        primaryPathway: 'RECOVER',
        pathways: ['RECOVER', 'BOND'],
        xp: baseXpForEvent('QUEST_RECOVER'),
        confidence: 0.78,
        href: '/activity',
        actionLabel: 'Log an easy session',
        safeStopEligible: false,
        personalRelevance: this.pathwayRelevance(learning, 'RECOVER'),
        score:
          0.48 +
          (1 - (coverage.get('RECOVER') ?? 0) / 100) * 0.24 +
          (learning.temporaryPace === 'easy' ? 0.06 : 0),
      },
      {
        key: 'favorite-ritual',
        title: 'Do one favorite thing together',
        description:
          'Pick a small ritual you both genuinely enjoy. Familiar can be as valuable as novel.',
        why: 'The Bond pathway is about shared fit, not performance.',
        primaryPathway: 'BOND',
        pathways: ['BOND'],
        xp: baseXpForEvent('QUEST_BOND'),
        confidence: 0.72,
        href: '/activity',
        actionLabel: 'Log the moment',
        safeStopEligible: false,
        personalRelevance: this.pathwayRelevance(learning, 'BOND'),
        score: 0.5 + (1 - (coverage.get('BOND') ?? 0) / 100) * 0.22,
      },
    ];

    const pool = [...candidates, ...templates]
      .filter(
        (candidate, index, items) => items.findIndex((item) => item.key === candidate.key) === index
      )
      .sort((a, b) => b.score - a.score);

    const selected: Candidate[] = [];
    for (const candidate of pool) {
      if (selected.length >= 3) break;
      if (selected.some((item) => item.primaryPathway === candidate.primaryPathway)) continue;
      selected.push(candidate);
    }
    for (const candidate of pool) {
      if (selected.length >= 3) break;
      if (!selected.includes(candidate)) selected.push(candidate);
    }

    const expiresAt = new Date(
      new Date(`${dateKey}T00:00:00.000Z`).getTime() + DAY_MS
    ).toISOString();
    return selected.slice(0, 3).map((candidate, index) => ({
      ...candidate,
      id: this.questId(userId, insights.pet.id, dateKey, candidate.key),
      variant: index === 0 ? 'recommended' : index === 2 ? 'wildcard' : 'alternative',
      expiresAt,
    }));
  }

  private pathwayRelevance(learning: AdventureLearningSignals, pathway: WellbeingPathway) {
    const durable = learning.durablePathwayPreference[pathway] ?? 0;
    const temporary = learning.temporaryPathwayModifier[pathway] ?? 0;
    return this.clamp(1 + durable + temporary, 0.9, 1.08);
  }

  private secondaryPathways(primary: WellbeingPathway): WellbeingPathway[] {
    const map: Record<WellbeingPathway, WellbeingPathway[]> = {
      MOVE: ['MOVE', 'BOND'],
      EXPLORE: ['EXPLORE', 'ENRICH', 'BOND'],
      ENRICH: ['ENRICH', 'BOND'],
      LEARN: ['LEARN', 'BOND'],
      CONNECT: ['CONNECT', 'BOND'],
      CARE: ['CARE', 'BOND'],
      RECOVER: ['RECOVER', 'BOND'],
      BOND: ['BOND'],
    };
    return map[primary];
  }

  private questDescription(pathway: WellbeingPathway, petName: string) {
    const copy: Record<WellbeingPathway, string> = {
      MOVE: `Choose movement that fits ${petName}'s current rhythm rather than chasing a distance target.`,
      EXPLORE: `Let ${petName} investigate the world at a comfortable pace.`,
      ENRICH: 'Add a short search, scent, puzzle, or foraging experience.',
      LEARN: 'Practice one reward-based skill with easy repetitions and clear exits.',
      CONNECT: 'Choose a social context with enough space, easy exits, and no pressure to finish.',
      CARE: 'Take one small preventive-care action that fits your existing care plan.',
      RECOVER: 'Choose calm movement, decompression, or rest and count it as real progress.',
      BOND: 'Do something that feels good for both halves of the team.',
    };
    return copy[pathway];
  }

  private questId(userId: string, petId: string, dateKey: string, key: string) {
    return createHash('sha256')
      .update(`${userId}:${petId}:${dateKey}:${key}`)
      .digest('hex')
      .slice(0, 24);
  }

  private clamp(value: number, min: number, max: number) {
    return Math.max(min, Math.min(max, value));
  }
}
