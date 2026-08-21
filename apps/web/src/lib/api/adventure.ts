import { apiClient } from './client';

export type WellbeingPathway =
  'MOVE' | 'EXPLORE' | 'ENRICH' | 'LEARN' | 'CONNECT' | 'CARE' | 'RECOVER' | 'BOND';

export type AdventureQuest = {
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

export type CompassPathway = {
  pathway: WellbeingPathway;
  label: string;
  recentDays: number;
  coverage: number;
  xp: number;
  lastEventAt: string | null;
};

export type AdventureDashboard = {
  pet: {
    id: string;
    name: string;
    species: string;
    avatarUrl?: string | null;
  };
  generatedAt: string;
  bondXp: number;
  rhythm: {
    activeWeeks: number;
    windowWeeks: number;
    label: string;
  };
  compass: CompassPathway[];
  quests: AdventureQuest[];
  learningSummary: string[];
  principles: string[];
  disclaimer: string;
};

export type CompleteQuestInput = {
  petId: string;
  dogExperience: 'loved_it' | 'comfortable' | 'not_their_thing';
  ownerExperience: 'great' | 'fine' | 'a_lot_today';
  safeOptOut?: boolean;
  memoryAssetId?: string;
  note?: string;
};

export type QuestCompletion = {
  reward: {
    careEventId: string;
    ledgerId: string | null;
    bondXp: number;
    pathway: WellbeingPathway;
    policyVersion: string;
    explanation: string;
    duplicate: boolean;
  };
  message: string;
};

export const adventureApi = {
  getMine: async (petId?: string) =>
    apiClient.get<AdventureDashboard>('/adventure/me', {
      params: petId ? { petId } : undefined,
    }) as unknown as Promise<AdventureDashboard>,

  selectQuest: async (questId: string, petId: string) =>
    apiClient.post(`/adventure/quests/${questId}/select`, { petId }) as unknown as Promise<{
      ok: boolean;
    }>,

  dismissQuest: async (questId: string, petId: string) =>
    apiClient.post(`/adventure/quests/${questId}/dismiss`, { petId }) as unknown as Promise<{
      ok: boolean;
    }>,

  completeQuest: async (questId: string, input: CompleteQuestInput) =>
    apiClient.post(
      `/adventure/quests/${questId}/complete`,
      input
    ) as unknown as Promise<QuestCompletion>,
};
