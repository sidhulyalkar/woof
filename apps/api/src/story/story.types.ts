export const STORY_SOURCE_TYPES = ['ACTIVITY', 'CARE_EVENT', 'MEDIA'] as const;
export type StorySourceType = (typeof STORY_SOURCE_TYPES)[number];

export type StoryCurationState = 'SAVED' | 'HIDDEN';

export type StoryCurationPayload = {
  schemaVersion: 'dogos-story-curation-v1';
  sourceType: StorySourceType;
  sourceId: string;
  state: StoryCurationState;
  note?: string;
  updatedAt: string;
};

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

export type StoryLifeStats = {
  activities: number;
  activeMinutes: number;
  distanceMeters: number;
  memories: number;
  namedPlaces: number;
  coverage: 'COMPLETE' | 'BOUNDED';
};
