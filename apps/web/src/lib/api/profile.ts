import { apiClient } from './client';
import type { AuthUser } from '../stores/auth-store';

export interface ProfileUpdate {
  handle?: string;
  bio?: string;
  avatarUrl?: string;
  visibility?: 'PUBLIC' | 'FRIENDS_ONLY' | 'PRIVATE';
}

export interface LegacyGamificationSummary {
  points?: number;
  completedAt?: string;
}

export const profileApi = {
  update: (data: ProfileUpdate) => apiClient.patch<AuthUser>('/users/me', data),
  gamificationSummary: () => apiClient.get<LegacyGamificationSummary>('/gamification/me/summary'),
};
