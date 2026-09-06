import apiClient from './client';
import { CreatePetDto, Pet } from '../types';

interface PetEnvelope {
  pets: Pet[];
  total: number;
  skip: number;
  take: number;
}

export type OwnedPet = {
  id: string;
  name: string;
  species: string;
  breed?: string | null;
  avatarUrl?: string | null;
  ownerId: string;
};

export type CreatedOwnedPet = OwnedPet & {
  householdMemberships: {
    householdId: string;
  }[];
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
  creationKey: string;
};

export const petsApi = {
  async getPets(ownerId?: string): Promise<PetEnvelope> {
    return apiClient.get('/pets', {
      params: ownerId ? { ownerId } : undefined,
    });
  },

  getMine: (take = 100) =>
    apiClient.get<OwnedPetsResponse>('/pets/me', {
      params: { take },
    }),

  /** Nearby discovery requires a dedicated privacy-preserving proximity API. */
  async getNearbyPets(
    _latitude: number,
    _longitude: number,
    _radiusMeters: number
  ): Promise<Pet[]> {
    return [];
  },

  async getPet(id: string): Promise<Pet> {
    return apiClient.get(`/pets/${id}`);
  },

  async createPet(data: CreatePetDto): Promise<Pet> {
    return apiClient.post('/pets', data);
  },

  createDog: (input: CreateDogInput) => apiClient.post<CreatedOwnedPet>('/pets', input),

  async updatePet(id: string, data: Partial<CreatePetDto> & { avatarUrl?: string }): Promise<Pet> {
    return apiClient.put(`/pets/${id}`, data);
  },

  async deletePet(id: string): Promise<void> {
    await apiClient.delete(`/pets/${id}`);
  },

  async uploadPetPhoto(id: string, photoUri: string): Promise<Pet> {
    const formData = new FormData();
    formData.append('file', {
      uri: photoUri,
      type: 'image/jpeg',
      name: 'pet.jpg',
    } as any);
    formData.append('folder', 'pets');

    const upload = await apiClient.post<{ url: string }>('/storage/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return apiClient.put(`/pets/${id}`, { avatarUrl: upload.url });
  },
};
