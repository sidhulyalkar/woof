import { apiClient } from './client';

export interface ProfileUpdate {
  handle?: string;
  bio?: string;
  avatarUrl?: string;
  visibility?: 'PUBLIC' | 'FRIENDS_ONLY' | 'PRIVATE';
}

export const profileApi = {
  update: (data: ProfileUpdate) => apiClient.patch('/users/me', data),
  gamificationSummary: () => apiClient.get('/gamification/me/summary'),
};
