import apiClient from './client';
import { Activity, CreateActivityDto, PaginatedResponse } from '../types';

export const activitiesApi = {
  async getActivities(page: number = 1, limit: number = 20): Promise<PaginatedResponse<Activity>> {
    return apiClient.get('/activities', { params: { page, limit } });
  },

  async getActivityById(id: string): Promise<Activity> {
    return apiClient.get(`/activities/${id}`);
  },

  async createActivity(data: CreateActivityDto): Promise<Activity> {
    return apiClient.post('/activities', data);
  },

  async updateActivity(id: string, data: Partial<CreateActivityDto>): Promise<Activity> {
    return apiClient.patch(`/activities/${id}`, data);
  },

  async deleteActivity(id: string): Promise<void> {
    return apiClient.delete(`/activities/${id}`);
  },

  async getMyActivities(petId?: string): Promise<Activity[]> {
    const params = petId ? { petId } : {};
    return apiClient.get('/activities/my-activities', { params });
  },

  async uploadActivityPhotos(activityId: string, photoUris: string[]): Promise<{ urls: string[] }> {
    const formData = new FormData();
    photoUris.forEach((uri, index) => {
      formData.append('files', {
        uri,
        type: 'image/jpeg',
        name: `activity-photo-${index}.jpg`,
      } as any);
    });

    return apiClient.post(`/activities/${activityId}/photos`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
};
