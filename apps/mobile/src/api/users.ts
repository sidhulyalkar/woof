import apiClient from './client';
import { User, UpdateUserDto, LeaderboardEntry } from '../types';

export const usersApi = {
  async getProfile(): Promise<User> {
    return apiClient.get('/users/profile');
  },

  async updateProfile(data: UpdateUserDto): Promise<User> {
    return apiClient.patch('/users/profile', data);
  },

  async uploadAvatar(photoUri: string): Promise<{ url: string }> {
    const formData = new FormData();
    formData.append('file', {
      uri: photoUri,
      type: 'image/jpeg',
      name: 'avatar.jpg',
    } as any);

    return apiClient.post('/users/avatar', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  async getUserById(id: string): Promise<User> {
    return apiClient.get(`/users/${id}`);
  },

  async searchUsers(query: string): Promise<User[]> {
    return apiClient.get('/users/search', { params: { q: query } });
  },
};
