import { CreatePetDto, Pet } from '../types';
import apiClient from './client';

interface PetEnvelope {
  pets: Pet[];
  total: number;
  skip: number;
  take: number;
}

export const petsApi = {
  async getPets(ownerId?: string): Promise<PetEnvelope> {
    return apiClient.get('/pets', {
      params: ownerId ? { ownerId } : undefined,
    });
  },

  /** Use the authenticated server-side owner boundary instead of listing every pet. */
  async getMyPets(): Promise<Pet[]> {
    const response = await apiClient.get<PetEnvelope>('/pets/me');
    return response.pets;
  },

  /**
   * Nearby-pet discovery needs a dedicated privacy-preserving proximity API.
   * Fail closed instead of approximating it by downloading all pet/owner records.
   */
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
