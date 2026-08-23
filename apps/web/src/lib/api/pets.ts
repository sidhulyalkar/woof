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

export type CreateDogInput = {
  name: string;
  species: 'DOG';
  breed?: string;
  sex?: 'MALE' | 'FEMALE' | 'UNKNOWN';
  birthdate?: string;
};

export const petsApi = {
  getMine: (take = 100) =>
    apiClient.get<OwnedPetsResponse>('/pets/me', {
      params: { take },
    }),

  createDog: (input: CreateDogInput) => apiClient.post<OwnedPet, CreateDogInput>('/pets', input),
};
