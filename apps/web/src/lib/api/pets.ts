import { apiClient } from './client';

export type OwnedPet = {
  id: string;
  name: string;
  species: string;
  breed?: string | null;
  avatarUrl?: string | null;
  createdAt: string;
  _count?: {
    activities: number;
    posts: number;
  };
};

export type OwnedPetsResponse = {
  pets: OwnedPet[];
  total: number;
  skip: number;
  take: number;
};

export const petsApi = {
  getMine: (take = 100) =>
    apiClient.get<OwnedPetsResponse>('/pets/me', {
      params: { take },
    }),
};
