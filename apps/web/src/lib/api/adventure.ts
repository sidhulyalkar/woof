import { getActivePetId } from '../active-pet';
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

function resolvedPetId(explicitPetId?: string) {
  return explicitPetId ?? getActivePetId() ?? undefined;
}

export const adventureApi = {
  getMine: (petId?: string) => {
    const activePetId = resolvedPetId(petId);
    return apiClient.get<AdventureDashboard>('/adventure/me', {
      params: activePetId ? { petId: activePetId } : undefined,
    });
  },

  selectQuest: (questId: string, petId: string) =>
    apiClient.post<{ ok: boolean }>(`/adventure/quests/${questId}/select`, { petId }),

  dismissQuest: (questId: string, petId: string) =>
    apiClient.post<{ ok: boolean }>(`/adventure/quests/${questId}/dismiss`, { petId }),

  completeQuest: (questId: string, input: CompleteQuestInput) =>
    apiClient.post<QuestCompletion>(`/adventure/quests/${questId}/complete`, input),
};
