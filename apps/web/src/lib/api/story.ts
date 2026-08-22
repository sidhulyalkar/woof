import { apiClient } from './client';

export type StorySourceType = 'ACTIVITY' | 'CARE_EVENT' | 'MEDIA';
export type StoryCurationState = 'SAVED' | 'HIDDEN';

export type StoryMoment = {
  id: string;
  sourceType: StorySourceType;
  sourceId: string;
  petIds: string[];
  petNames: string[];
  occurredAt: string;
  kind: string;
  title: string;
  summary: string;
  pathway?: string;
  mediaType?: string;
  favorite?: boolean;
  suggested: boolean;
  curation: {
    state: StoryCurationState | null;
    note: string | null;
  };
};

export type StoryMilestone = {
  id: string;
  title: string;
  description: string;
  achievedAt: string;
};

export type StoryDashboard = {
  moments: StoryMoment[];
  milestones: StoryMilestone[];
  stats: {
    activities: number;
    activeMinutes: number;
    distanceMeters: number;
    memories: number;
    namedPlaces: number;
    coverage: 'COMPLETE' | 'BOUNDED';
  };
  nextBefore: string | null;
  principles: string[];
};

export type StoryQuery = {
  petId?: string;
  before?: string;
  limit?: number;
};

export type StoryCurationInput = {
  sourceType: StorySourceType;
  sourceId: string;
  action: 'SAVE' | 'CLEAR';
  note?: string;
};

export const storyApi = {
  get: (query: StoryQuery = {}) =>
    apiClient.get<StoryDashboard>('/story', {
      params: Object.keys(query).length ? query : undefined,
    }),

  curate: (input: StoryCurationInput) =>
    apiClient.put<
      {
        sourceType: StorySourceType;
        sourceId: string;
        state: StoryCurationState | null;
        note: string | null;
      },
      StoryCurationInput
    >('/story/curation', input),
};
