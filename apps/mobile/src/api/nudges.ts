import apiClient from './client';
import { Nudge } from '../types';

export const nudgesApi = {
  async getNudges(): Promise<Nudge[]> {
    return apiClient.get('/nudges');
  },

  async dismissNudge(id: string): Promise<void> {
    return apiClient.patch(`/nudges/${id}/dismiss`);
  },

  async acceptNudge(id: string): Promise<void> {
    return apiClient.patch(`/nudges/${id}/accept`);
  },
};
