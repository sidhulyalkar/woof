import { NotFoundException } from '@nestjs/common';
import { CareEventsService } from '../care-events/care-events.service';
import type { RewardReceipt } from '../care-events/care-event.types';
import { InsightsService } from '../insights/insights.service';
import { PrismaService } from '../prisma/prisma.service';
import { AdventureService } from './adventure.service';
import type { CompleteQuestDto } from './dto/adventure.dto';

const quest = {
  id: 'quest-1',
  key: 'skill-spark',
  title: 'Five-minute skill spark',
  description: 'Practice one easy skill.',
  why: 'Short comfortable repetitions build communication.',
  primaryPathway: 'LEARN' as const,
  pathways: ['LEARN', 'BOND'] as const,
  xp: 15,
  confidence: 0.82,
  href: '/coach',
  actionLabel: 'Open Coach',
  variant: 'recommended' as const,
  safeStopEligible: true,
  personalRelevance: 1.02,
  expiresAt: '2026-08-22T00:00:00.000Z',
};

const dashboard = {
  pet: { id: 'pet-1', name: 'Shasta', species: 'DOG', avatarUrl: null },
  generatedAt: '2026-08-21T12:00:00.000Z',
  bondXp: 12,
  rhythm: { activeWeeks: 2, windowWeeks: 5, label: 'A steady rhythm is growing' },
  compass: [],
  quests: [{ ...quest, pathways: [...quest.pathways] }],
  learningSummary: [],
  principles: [],
  disclaimer: 'Opportunity coverage only.',
};

const dto = (overrides: Partial<CompleteQuestDto> = {}): CompleteQuestDto => ({
  petId: 'pet-1',
  dogExperience: 'comfortable',
  ownerExperience: 'fine',
  ...overrides,
});

const receipt = (overrides: Partial<RewardReceipt> = {}): RewardReceipt => ({
  careEventId: 'care-1',
  ledgerId: 'ledger-1',
  bondXp: 17,
  pathway: 'LEARN',
  policyVersion: 'bond-xp-v1',
  explanation: 'trusted reward',
  duplicate: false,
  ...overrides,
});

type PrismaHarness = {
  mediaAsset: { findFirst: jest.Mock };
  telemetry: { create: jest.Mock };
};

type CareEventsHarness = {
  record: jest.Mock;
  recordQuestInteraction: jest.Mock;
  getRecentSelectedQuestContext: jest.Mock;
};

describe('AdventureService', () => {
  let prismaHarness: PrismaHarness;
  let careHarness: CareEventsHarness;
  let service: AdventureService;

  beforeEach(() => {
    prismaHarness = {
      mediaAsset: { findFirst: jest.fn().mockResolvedValue(null) },
      telemetry: { create: jest.fn().mockResolvedValue({ id: 'telemetry-1' }) },
    };
    careHarness = {
      record: jest.fn().mockResolvedValue(receipt()),
      recordQuestInteraction: jest.fn().mockResolvedValue({ id: 'interaction-1' }),
      getRecentSelectedQuestContext: jest.fn().mockResolvedValue(null),
    };

    service = new AdventureService(
      prismaHarness as unknown as PrismaService,
      {} as InsightsService,
      careHarness as unknown as CareEventsService
    );
    jest.spyOn(service, 'getDashboard').mockResolvedValue(dashboard);
  });

  it('stores only completion semantics when a quest is selected', async () => {
    await service.recordInteraction('user-1', 'pet-1', quest.id, 'SELECTED');

    expect(careHarness.recordQuestInteraction).toHaveBeenCalledWith(
      expect.objectContaining({
        interaction: 'SELECTED',
        pathway: 'LEARN',
        context: expect.objectContaining({
          learningPolicyVersion: 'adventure-learning-v2',
          questSnapshot: {
            id: quest.id,
            key: quest.key,
            title: quest.title,
            primaryPathway: 'LEARN',
            safeStopEligible: true,
            personalRelevance: quest.personalRelevance,
          },
        }),
      })
    );

    const call = careHarness.recordQuestInteraction.mock.calls[0]?.[0] as {
      context: { questSnapshot: Record<string, unknown> };
    };
    expect(call.context.questSnapshot).not.toHaveProperty('xp');
    expect(call.context.questSnapshot).not.toHaveProperty('href');
  });

  it('turns an eligible safe opt-out into Bond rather than Learn credit', async () => {
    careHarness.record.mockResolvedValue(receipt({ pathway: 'BOND', bondXp: 18 }));

    await service.completeQuest(
      'user-1',
      quest.id,
      dto({ dogExperience: 'not_their_thing', ownerExperience: 'a_lot_today', safeOptOut: true })
    );

    expect(careHarness.record).toHaveBeenCalledWith(
      expect.objectContaining({
        eventType: 'SAFE_OPT_OUT',
        pathway: 'BOND',
        safetyEligible: true,
        context: expect.objectContaining({ originalPathway: 'LEARN' }),
      })
    );
    expect(careHarness.recordQuestInteraction).toHaveBeenCalledWith(
      expect.objectContaining({
        interaction: 'COMPLETED',
        pathway: 'BOND',
        context: expect.objectContaining({
          originalPathway: 'LEARN',
          rewardPathway: 'BOND',
          learningPolicyVersion: 'adventure-learning-v2',
        }),
      })
    );
    expect(prismaHarness.telemetry.create).toHaveBeenCalledWith(
      expect.objectContaining({
        data: expect.objectContaining({
          data: expect.objectContaining({
            pathway: 'BOND',
            originalPathway: 'LEARN',
            rewardPathway: 'BOND',
            learningPolicyVersion: 'adventure-learning-v2',
          }),
        }),
      })
    );
  });

  it('treats not-their-thing as Bond learning instead of inflating the attempted pathway', async () => {
    careHarness.record.mockResolvedValue(receipt({ pathway: 'BOND', bondXp: 12 }));

    await service.completeQuest(
      'user-1',
      quest.id,
      dto({ dogExperience: 'not_their_thing', safeOptOut: false })
    );

    expect(careHarness.record).toHaveBeenCalledWith(
      expect.objectContaining({
        eventType: 'QUEST_BOND',
        pathway: 'BOND',
        context: expect.objectContaining({ originalPathway: 'LEARN' }),
      })
    );
    expect(careHarness.recordQuestInteraction).toHaveBeenCalledWith(
      expect.objectContaining({
        interaction: 'COMPLETED',
        pathway: 'BOND',
        context: expect.objectContaining({
          originalPathway: 'LEARN',
          rewardPathway: 'BOND',
          learningPolicyVersion: 'adventure-learning-v2',
        }),
      })
    );
  });

  it('does not grant a memory modifier for an unverified media id', async () => {
    await service.completeQuest('user-1', quest.id, dto({ memoryAssetId: 'unowned-asset' }));

    expect(prismaHarness.mediaAsset.findFirst).toHaveBeenCalledWith({
      where: {
        id: 'unowned-asset',
        ownerId: 'user-1',
        petId: 'pet-1',
        status: 'READY',
      },
      select: { id: true },
    });
    expect(careHarness.record).toHaveBeenCalledWith(
      expect.objectContaining({
        context: expect.objectContaining({ memoryAdded: false, memoryAssetId: null }),
      })
    );
  });

  it('allows the small memory modifier only for READY media owned by the same dog-owner pair', async () => {
    prismaHarness.mediaAsset.findFirst.mockResolvedValue({ id: 'owned-asset' });

    await service.completeQuest('user-1', quest.id, dto({ memoryAssetId: 'owned-asset' }));

    expect(careHarness.record).toHaveBeenCalledWith(
      expect.objectContaining({
        context: expect.objectContaining({ memoryAdded: true, memoryAssetId: 'owned-asset' }),
      })
    );
  });

  it('completes a recently selected quest after it reshuffles out of the current deck', async () => {
    jest.spyOn(service, 'getDashboard').mockResolvedValue({ ...dashboard, quests: [] });
    careHarness.getRecentSelectedQuestContext.mockResolvedValue({
      created_at: new Date(),
      context: {
        questSnapshot: {
          id: quest.id,
          key: quest.key,
          title: quest.title,
          primaryPathway: quest.primaryPathway,
          safeStopEligible: quest.safeStopEligible,
          personalRelevance: quest.personalRelevance,
        },
      },
    });

    await service.completeQuest('user-1', quest.id, dto());

    expect(careHarness.getRecentSelectedQuestContext).toHaveBeenCalledWith(
      'user-1',
      'pet-1',
      quest.id
    );
    expect(careHarness.record).toHaveBeenCalledWith(
      expect.objectContaining({ eventType: 'QUEST_LEARN', pathway: 'LEARN' })
    );
  });

  it('rejects a malformed selected snapshot instead of trusting persisted JSON blindly', async () => {
    jest.spyOn(service, 'getDashboard').mockResolvedValue({ ...dashboard, quests: [] });
    careHarness.getRecentSelectedQuestContext.mockResolvedValue({
      created_at: new Date(),
      context: {
        questSnapshot: {
          id: quest.id,
          key: quest.key,
          title: quest.title,
          primaryPathway: 'NOT_A_PATHWAY',
          safeStopEligible: true,
          personalRelevance: 99,
        },
      },
    });

    await expect(service.completeQuest('user-1', quest.id, dto())).rejects.toBeInstanceOf(
      NotFoundException
    );
    expect(careHarness.record).not.toHaveBeenCalled();
  });

  it('does not report failure after the authoritative reward commits if telemetry fails', async () => {
    careHarness.recordQuestInteraction.mockRejectedValue(new Error('interaction store offline'));
    prismaHarness.telemetry.create.mockRejectedValue(new Error('telemetry store offline'));

    await expect(service.completeQuest('user-1', quest.id, dto())).resolves.toEqual(
      expect.objectContaining({ reward: expect.objectContaining({ careEventId: 'care-1' }) })
    );
  });

  it('uses an idempotent retry to repair interaction telemetry without implying new XP', async () => {
    careHarness.record.mockResolvedValue(receipt({ duplicate: true, bondXp: 17 }));

    const result = await service.completeQuest('user-1', quest.id, dto());

    expect(result.reward.duplicate).toBe(true);
    expect(careHarness.recordQuestInteraction).toHaveBeenCalledWith(
      expect.objectContaining({ interaction: 'COMPLETED' })
    );
    expect(prismaHarness.telemetry.create).toHaveBeenCalledWith(
      expect.objectContaining({
        data: expect.objectContaining({ data: expect.objectContaining({ duplicate: true }) }),
      })
    );
  });
});
