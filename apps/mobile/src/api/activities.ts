import { Activity, CreateActivityDto, PaginatedResponse } from '../types';
import type { CreateActivityRequest, UpdateActivityRequest } from '../types/activity';
import apiClient from './client';

type MobileActivityWrite = CreateActivityRequest | CreateActivityDto;

type ActivityListResponse = {
  activities: Activity[];
  total: number;
  skip: number;
  take: number;
};

function isLegacyActivityWrite(data: MobileActivityWrite): data is CreateActivityDto {
  return 'startTime' in data || 'duration' in data || 'title' in data;
}

/**
 * Keep the currently shipped mobile form contract working while all new dogOS
 * surfaces move to the canonical server contract. This translation happens on
 * the client boundary so the API can stay strict (`forbidNonWhitelisted`) and
 * old apps do not need a synchronized release.
 */
export function normalizeActivityWrite(data: MobileActivityWrite): CreateActivityRequest {
  if (!isLegacyActivityWrite(data)) return data;

  const startedAt = data.startTime;
  const endedAt =
    data.endTime ??
    (data.duration > 0
      ? new Date(new Date(startedAt).getTime() + data.duration * 60_000).toISOString()
      : undefined);

  const location = data.location
    ? {
        latitude: data.location.latitude,
        longitude: data.location.longitude,
        ...(data.location.address ? { address: data.location.address } : {}),
      }
    : undefined;

  return {
    petId: data.petId,
    type: data.type,
    startedAt,
    endedAt,
    ...(location ? { route: { start: location } } : {}),
    humanMetrics: {
      durationMinutes: data.duration,
      ...(data.distance !== undefined ? { distanceMeters: data.distance } : {}),
      ...(data.title ? { legacyTitle: data.title } : {}),
      ...(data.description ? { legacyDescription: data.description } : {}),
    },
  };
}

export const activitiesApi = {
  async getActivities(page: number = 1, limit: number = 20): Promise<PaginatedResponse<Activity>> {
    const safePage = Math.max(1, page);
    const safeLimit = Math.max(1, limit);
    const skip = (safePage - 1) * safeLimit;
    const response = await apiClient.get<ActivityListResponse>('/activities', {
      params: { skip, take: safeLimit },
    });

    return {
      data: response.activities,
      total: response.total,
      page: safePage,
      limit: response.take,
      hasMore: response.skip + response.activities.length < response.total,
    };
  },

  async getActivityById(id: string): Promise<Activity> {
    return apiClient.get(`/activities/${id}`);
  },

  async createActivity(data: MobileActivityWrite): Promise<Activity> {
    return apiClient.post('/activities', normalizeActivityWrite(data));
  },

  async updateActivity(
    id: string,
    data: UpdateActivityRequest | Partial<CreateActivityDto>
  ): Promise<Activity> {
    const payload = isLegacyActivityWrite(data as MobileActivityWrite)
      ? normalizeActivityWrite(data as CreateActivityDto)
      : data;
    return apiClient.put(`/activities/${id}`, payload);
  },

  async deleteActivity(id: string): Promise<void> {
    return apiClient.delete(`/activities/${id}`);
  },

  async getMyActivities(petId?: string): Promise<Activity[]> {
    const response = await apiClient.get<ActivityListResponse>('/activities', {
      params: { ...(petId ? { petId } : {}), skip: 0, take: 100 },
    });
    return response.activities;
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
