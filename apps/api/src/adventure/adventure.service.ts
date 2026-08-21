import { Injectable, NotFoundException } from '@nestjs/common';
import { createHash, randomUUID } from 'node:crypto';
import { CareEventsService } from '../care-events/care-events.service';
import {
  QUEST_EVENT_TYPES,
  type CareSummary,
  type WellbeingPathway,
} from '../care-events/care-event.types';
import { baseXpForEvent } from '../care-events/reward-policy';
import { InsightsService } from '../insights/insights.service';
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
  constructor(
    private readonly prisma: PrismaService,
    private readonly insights: InsightsService,
    private readonly careEvents: CareEventsService,
  ) {}

  async getDashboard(userId: string, requestedPetId?: string) {
    const insights = await this.insights.getForUser(userId, requestedPetId);
    const summary = await this.careEvents.getSummary(userId, insights.pet.id);
    const quests = this.buildQuests(userId, insights, summary);

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
    interaction: 'SELECTED' | 'DISMISSED',
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
      context: { questKey: quest.key, confidence: quest.confidence },
    });
    return { ok: true };
  }

  async completeQuest(userId: string, questId: string, dto: CompleteQuestDto) {
    const dashboard = await this.getDashboard(userId, dto.petId);
    const quest = dashboard.quests.find((candidate) => candidate.id === questId);
    if (!quest) throw new NotFoundException('This quest is no longer available');

    const safeOptOut = Boolean(dto.safeOptOut && quest.safeStopEligible);
    const learnedMismatch = dto.dogExperience === 'not_their_thing' && !safeOptOut;
    const eventType = safeOptOut
      ? 'SAFE_OPT_OUT'
      : learnedMismatch
        ? QUEST_EVENT_TYPES.BOND
        : QUEST_EVENT_TYPES[quest.primaryPathway];
    const rewardPathway: WellbeingPathway = safeOptOut || learnedMismatch ? 'BOND' : quest.primaryPathway;

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
        personalRelevance: quest.personalRelevance,
        newPlace: dto.newPlace ?? false,
        memoryAdded: Boolean(dto.memoryAssetId),
        memoryAssetId: dto.memoryAssetId ?? null,
      },
      outcome: {
        dogExperience: dto.dogExperience,
        ownerExperience: dto.ownerExperience,
        safeOptOut,
        note: dto.note ?? null,
      },
    });

    if (!receipt.duplicate) {
      await this.careEvents.recordQuestInteraction({
        userId,
        petId: dto.petId,
        questId,
        interaction: 'COMPLETED',
        pathway: rewardPathway,
        context: {
          questKey: quest.key,
          dogExperience: dto.dogExperience,
          ownerExperience: dto.ownerExperience,
          safeOptOut,
          bondXp: receipt.bondXp,
        },
      });
    }

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
          bondXp: receipt.bondXp,
          dogExperience: dto.dogExperience,
          ownerExperience: dto.ownerExperience,
        },
      },
    });

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

  private buildQuests(
    userId: string,
    insights: Awaited<ReturnType<InsightsService['getForUser']>>,
    summary: CareSummary,
  ): Quest[] {
    const dateKey = new Date().toISOString().slice(0, 10);
    const coverage = new Map(summary.pathways.map((item) => [item.pathway, item.coverage]));
    const preference = this.preferenceSignals(summary);

    const candidates: Candidate[] = insights.recommendations.map((recommendation) => {
      const pathway = CATEGORY_PATHWAYS[recommendation.category] ?? 'BOND';
      const eventType = QUEST_EVENT_TYPES[pathway];
      const gap = 1 - (coverage.get(pathway) ?? 0) / 100;
      const personalRelevance = this.clamp(1 + (preference[pathway] ?? 0), 0.9, 1.08);
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
        personalRelevance: this.clamp(1 + (preference.EXPLORE ?? 0), 0.9, 1.08),
        score: 0.62 + (1 - (coverage.get('EXPLORE') ?? 0) / 100) * 0.28,
      },
      {
        key: 'skill-spark',
        title: 'Five-minute skill spark',
        description: 'Choose one tiny reward-based skill and stop while the session still feels easy.',
        why: 'Short comfortable repetitions build communication without making training a grind.',
        primaryPathway: 'LEARN',
        pathways: ['LEARN', 'BOND'],
        xp: baseXpForEvent('QUEST_LEARN'),
        confidence: 0.82,
        href: '/coach',
        actionLabel: 'Open Coach',
        safeStopEligible: true,
        personalRelevance: this.clamp(1 + (preference.LEARN ?? 0), 0.9, 1.08),
        score: 0.58 + (1 - (coverage.get('LEARN') ?? 0) / 100) * 0.3,
      },
      {
        key: 'easy-day',
        title: 'Choose an easy day',
        description: 'Keep the outing simple, add calm sniffing or enrichment, and leave room for decompression.',
        why: 'Recovery is a valid wellbeing action. Woof does not treat more miles as automatically better.',
        primaryPathway: 'RECOVER',
        pathways: ['RECOVER', 'BOND'],
        xp: baseXpForEvent('QUEST_RECOVER'),
        confidence: 0.78,
        href: '/activity',
        actionLabel: 'Log an easy session',
        safeStopEligible: false,
        personalRelevance: this.clamp(1 + (preference.RECOVER ?? 0), 0.9, 1.08),
        score: 0.48 + (1 - (coverage.get('RECOVER') ?? 0) / 100) * 0.24,
      },
      {
        key: 'favorite-ritual',
        title: 'Do one favorite thing together',
        description: 'Pick a small ritual you both genuinely enjoy. Familiar can be as valuable as novel.',
        why: 'The Bond pathway is about shared fit, not performance.',
        primaryPathway: 'BOND',
        pathways: ['BOND'],
        xp: baseXpForEvent('QUEST_BOND'),
        confidence: 0.72,
        href: '/activity',
        actionLabel: 'Log the moment',
        safeStopEligible: false,
        personalRelevance: this.clamp(1 + (preference.BOND ?? 0), 0.9, 1.08),
        score: 0.5 + (1 - (coverage.get('BOND') ?? 0) / 100) * 0.22,
      },
    ];

    const pool = [...candidates, ...templates]
      .filter((candidate, index, items) => items.findIndex((item) => item.key === candidate.key) === index)
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

    const expiresAt = new Date(new Date(`${dateKey}T00:00:00.000Z`).getTime() + DAY_MS).toISOString();
    return selected.slice(0, 3).map((candidate, index) => ({
      ...candidate,
      id: this.questId(userId, insights.pet.id, dateKey, candidate.key),
      variant: index === 0 ? 'recommended' : index === 2 ? 'wildcard' : 'alternative',
      expiresAt,
    }));
  }

  private preferenceSignals(summary: CareSummary) {
    const signals: Partial<Record<WellbeingPathway, number>> = {};
    for (const event of summary.recentEvents.slice(0, 10)) {
      const dog = event.outcome?.dogExperience;
      const owner = event.outcome?.ownerExperience;
      const positiveDog = dog === 'loved_it' || dog === 'comfortable';
      const positiveOwner = owner === 'great' || owner === 'fine';
      const negative = dog === 'not_their_thing' || owner === 'a_lot_today';
      const delta = negative ? -0.025 : positiveDog && positiveOwner ? 0.018 : 0;
      signals[event.pathway] = this.clamp((signals[event.pathway] ?? 0) + delta, -0.1, 0.08);
    }
    return signals;
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
