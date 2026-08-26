import { apiClient } from './client';

export type OwnedPet = {
  id: string;
  name: string;
  species: string;
  breed?: string | null;
  birthdate?: string | null;
  avatarUrl?: string | null;
  createdAt: string;
  _count?: {
    activities: number;
    posts: number;
  };
};

export type CreatedOwnedPet = OwnedPet & {
  householdMemberships: Array<{
    householdId: string;
  }>;
};

export type OwnedPetsResponse = {
  pets: OwnedPet[];
  total: number;
  skip: number;
  take: number;
};

export type CreatePetInput = {
  name: string;
  species: string;
  breed?: string;
  sex?: 'MALE' | 'FEMALE' | 'UNKNOWN';
  birthdate?: string;
  creationKey?: string;
};

export type CreateDogInput = CreatePetInput & {
  species: 'DOG';
};

export type UpdatePetInput = Partial<
  Pick<CreatePetInput, 'name' | 'species' | 'breed' | 'sex' | 'birthdate'> & {
    avatarUrl: string;
  }
>;

export const petsApi = {
  getMine: (take = 100) =>
    apiClient.get<OwnedPetsResponse>('/pets/me', {
      params: { take },
    }),

  createPet: (input: CreatePetInput) =>
    apiClient.post<CreatedOwnedPet, CreatePetInput>('/pets', input),

  createDog: (input: CreateDogInput) =>
    apiClient.post<CreatedOwnedPet, CreateDogInput>('/pets', input),

  updatePet: (petId: string, input: UpdatePetInput) =>
    apiClient.put<OwnedPet, UpdatePetInput>(`/pets/${encodeURIComponent(petId)}`, input),
};
