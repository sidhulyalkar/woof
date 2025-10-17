import apiClient from './client';
import { Notification, PaginatedResponse } from '../types';

export const notificationsApi = {
  async getNotifications(page: number = 1, limit: number = 20): Promise<PaginatedResponse<Notification>> {
    return apiClient.get('/notifications', { params: { page, limit } });
  },

  async markAsRead(id: string): Promise<void> {
    return apiClient.patch(`/notifications/${id}/read`);
  },

  async markAllAsRead(): Promise<void> {
    return apiClient.post('/notifications/read-all');
  },

  async registerPushToken(token: string, platform: 'ios' | 'android'): Promise<void> {
    return apiClient.post('/notifications/push-token', { token, platform });
  },
};
