import apiClient from './client';
import { LeaderboardEntry, Badge } from '../types';

export const gamificationApi = {
  async getLeaderboard(timeframe: 'weekly' | 'monthly' | 'all-time' = 'all-time'): Promise<LeaderboardEntry[]> {
    return apiClient.get('/gamification/leaderboard', { params: { timeframe } });
  },

  async getMyBadges(): Promise<Badge[]> {
    return apiClient.get('/gamification/badges');
  },

  async getMyStats(): Promise<{
    points: number;
    level: number;
    rank: number;
    activitiesCount: number;
    badges: Badge[];
  }> {
    return apiClient.get('/gamification/stats');
  },
};
