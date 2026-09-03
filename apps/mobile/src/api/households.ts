import apiClient from './client';

export type HouseholdPet = {
  id: string;
  status: string;
  joinedAt: string;
  pet: {
    id: string;
    name: string;
    species: string;
    breed?: string | null;
    birthdate?: string | null;
    avatarUrl?: string | null;
  };
};

export type HouseholdSnapshot = {
  id: string;
  name: string;
  timezone?: string | null;
  viewerRole: string;
  pets: HouseholdPet[];
};

export const householdsApi = {
  getMine: () => apiClient.get<HouseholdSnapshot[]>('/households/me'),
};
