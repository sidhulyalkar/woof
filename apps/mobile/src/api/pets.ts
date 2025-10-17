import apiClient from './client';
import { Pet, CreatePetDto, PaginatedResponse } from '../types';

export const petsApi = {
  async getMyPets(): Promise<Pet[]> {
    return apiClient.get('/pets/my-pets');
  },

  async getPetById(id: string): Promise<Pet> {
    return apiClient.get(`/pets/${id}`);
  },

  async createPet(data: CreatePetDto): Promise<Pet> {
    return apiClient.post('/pets', data);
  },

  async updatePet(id: string, data: Partial<CreatePetDto>): Promise<Pet> {
    return apiClient.patch(`/pets/${id}`, data);
  },

  async deletePet(id: string): Promise<void> {
    return apiClient.delete(`/pets/${id}`);
  },

  async uploadPetPhoto(petId: string, photoUri: string): Promise<{ url: string }> {
    const formData = new FormData();
    formData.append('file', {
      uri: photoUri,
      type: 'image/jpeg',
      name: 'pet-photo.jpg',
    } as any);

    return apiClient.post(`/pets/${petId}/photo`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  async getNearbyPets(latitude: number, longitude: number, radius: number = 5000): Promise<Pet[]> {
    return apiClient.get('/pets/nearby', {
      params: { latitude, longitude, radius },
    });
  },
};
