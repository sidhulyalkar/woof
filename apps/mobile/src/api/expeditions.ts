import apiClient from './client';

export type ExpeditionScope = 'GLOBAL' | 'PACK';
export type ExpeditionObjectiveKey = 'SNIFF_EXPLORE' | 'RECOVERY_COUNTS' | 'READ_THE_ROOM';
export type ExpeditionSourceType = 'CARE_EVENT' | 'HUMAN_SKILL_ATTEMPT';

export type ExpeditionObjective = {
  key: ExpeditionObjectiveKey;
  title: string;
  description: string;
  sourceType: ExpeditionSourceType;
  categories: string[];
  total: number;
  contributors: number;
  myContribution: number;
  cap: {
    perContributor: number;
    perCategory: number;
  };
  target: null;
  status: 'CALIBRATING';
};

export type ExpeditionProjection = {
  expeditionKey: string;
  expeditionVersion: string;
  policyVersion: string;
  scope: ExpeditionScope;
  pack?: {
    id: string;
    name: string;
    memberCount: number;
  };
  season: {
    key: string;
    startsAt: string;
    endsAt: string;
  };
  generatedAt: string;
  objectives: ExpeditionObjective[];
  principles: string[];
};

export type ExpeditionJournalEntry = {
  season: {
    key: string;
    startsAt: string;
    endsAt: string;
  };
  state: 'ACTIVE' | 'PAST';
  landmarks: Array<{
    key: ExpeditionObjectiveKey;
    title: string;
  }>;
};

export type ExpeditionJournal = {
  expeditionKey: string;
  expeditionVersion: string;
  policyVersion: string;
  scope: 'GLOBAL';
  generatedAt: string;
  coverage: {
    kind: 'RECENT_PARTICIPATED_SEASONS';
    maxSeasons: number;
  };
  entries: ExpeditionJournalEntry[];
  principles: string[];
  disclaimer: string;
};

export const expeditionApi = {
  global: () => apiClient.get<ExpeditionProjection>('/expeditions/global'),
  journal: () => apiClient.get<ExpeditionJournal>('/expeditions/journal'),
  pack: (packId: string) => apiClient.get<ExpeditionProjection>(`/expeditions/packs/${packId}`),
};
