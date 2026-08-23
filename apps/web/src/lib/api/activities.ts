import { apiClient } from './client';

export type ActivityPet = {
  id: string;
  name: string;
  species: string;
  avatarUrl?: string | null;
};

export type CanonicalActivity = {
  id: string;
  userId: string;
  petId?: string | null;
  householdId?: string | null;
  startedAt: string;
  endedAt?: string | null;
  type: string;
  route?: unknown;
  humanMetrics?: unknown;
  petMetrics?: unknown;
  jointMetrics?: unknown;
  pet?: ActivityPet | null;
  petParticipants: Array<{
    petId: string;
    metrics?: unknown;
    pet: ActivityPet;
  }>;
  _count?: {
    posts: number;
  };
};

export type ActivitiesResponse = {
  activities: CanonicalActivity[];
  total: number;
  skip: number;
  take: number;
};

export type CreateActivityInput = {
  petIds: string[];
  startedAt: string;
  endedAt: string;
  type: string;
  jointMetrics?: Record<string, unknown>;
};

export const activitiesApi = {
  getMine: (input: { petId?: string; skip?: number; take?: number } = {}) =>
    apiClient.get<ActivitiesResponse>('/activities', {
      params: {
        ...(input.petId ? { petId: input.petId } : {}),
        skip: input.skip ?? 0,
        take: input.take ?? 25,
      },
    }),

  create: (input: CreateActivityInput) =>
    apiClient.post<CanonicalActivity, CreateActivityInput>('/activities', input),
};
