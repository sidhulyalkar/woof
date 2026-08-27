import { CareEventsService } from '../care-events/care-events.service';
import {
  WELLBEING_PATHWAYS,
  type AdventureLearningCareEvent,
  type CareSummary,
} from '../care-events/care-event.types';
import { InsightsService } from '../insights/insights.service';
import { PrismaService } from '../prisma/prisma.service';
import { AdventureService } from './adventure.service';

type RankedQuest = {
  key: string;
  primaryPathway: string;
  personalRelevance: number;
};

type QuestBuilder = {
  buildQuests: (
    userId: string,
    insights: object,
    summary: CareSummary,
    learningEvents: AdventureLearningCareEvent[]
  ) => RankedQuest[];
};

const insights = {
  pet: { id: 'pet-1', name: 'Shasta' },
  recommendations: [],
  algorithm: { confidence: 0.7 },
};

function summary(): CareSummary {
  return {
    bondXp: 0,
    rhythm: { activeWeeks: 0, windowWeeks: 5, label: 'Every useful moment can start a rhythm' },
    pathways: WELLBEING_PATHWAYS.map((pathway) => ({
      pathway,
      label: pathway,
      recentDays: 0,
      coverage: pathway === 'BOND' ? 68.18 : 100,
      xp: 0,
      lastEventAt: null,
    })),
    recentEvents: [],
  };
}

describe('Adventure learning ranking integration', () => {
  it('lets original-pathway mismatch change ranking without punishing the BOND reward pathway', () => {
    const service = new AdventureService(
      {} as PrismaService,
      {} as InsightsService,
      {} as CareEventsService
    );
    const buildQuests = (service as unknown as QuestBuilder).buildQuests.bind(service);

    const neutral = buildQuests('user-1', insights, summary(), []);
    const mismatch = buildQuests('user-1', insights, summary(), [
      {
        id: 'mismatch-1',
        eventType: 'QUEST_BOND',
        pathway: 'BOND',
        occurredAt: new Date().toISOString(),
        context: { originalPathway: 'LEARN' },
        outcome: {
          dogExperience: 'not_their_thing',
          ownerExperience: 'fine',
          safeOptOut: false,
        },
      },
    ]);

    expect(neutral.map((quest) => quest.key)).toEqual([
      'sniffari',
      'skill-spark',
      'favorite-ritual',
    ]);
    expect(mismatch.map((quest) => quest.key)).toEqual([
      'sniffari',
      'favorite-ritual',
      'skill-spark',
    ]);

    const learnedSkill = mismatch.find((quest) => quest.key === 'skill-spark');
    const bondRitual = mismatch.find((quest) => quest.key === 'favorite-ritual');
    expect(learnedSkill?.personalRelevance).toBeLessThan(1);
    expect(bondRitual?.personalRelevance).toBe(1);
  });
});
